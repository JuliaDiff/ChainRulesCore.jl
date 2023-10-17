"""
    (p::ProjectTo{T})(dx)

Projects the tangent `dx` onto a specific tangent space.

The type `T` is meant to encode the largest acceptable space, so usually
this enforces `p(dx)::T`. But some subspaces which aren't subtypes of `T` may
be allowed, and in particular `dx::AbstractZero` always passes through.

Usually `T` is the "outermost" part of the type, and `p` stores additional
properties such as projectors for each constituent field.
Arrays have either one projector `p.element` expressing the element type for
an array of numbers, or else an array of projectors `p.elements`.
These properties can be supplied as keyword arguments on construction,
`p = ProjectTo{T}(; field=data, element=Projector(x))`. For each `T` in use,
corresponding methods should be written for `ProjectTo{T}(dx)` with nonzero `dx`.

When called on `dx::Thunk`, the projection is inserted into the thunk.
"""
struct ProjectTo{P,D<:NamedTuple}
    info::D
end
ProjectTo{P}(info::D) where {P,D<:NamedTuple} = ProjectTo{P,D}(info)

# We'd like to write
# ProjectTo{P}(; kwargs...) where {P} = ProjectTo{P}(NamedTuple(kwargs))
#
# but the kwarg dispatcher has non-trivial complexity. See rules.jl for an
# explanation of this trick.
const EMPTY_NT = NamedTuple()
ProjectTo{P}() where {P} = ProjectTo{P}(EMPTY_NT)

const Type_kwfunc = Core.kwftype(Type).instance
function (::typeof(Type_kwfunc))(kws::Any, ::Type{ProjectTo{P}}) where {P}
    return ProjectTo{P}(NamedTuple(kws))
end

Base.getproperty(p::ProjectTo, name::Symbol) = getproperty(backing(p), name)
Base.propertynames(p::ProjectTo) = propertynames(backing(p))
backing(project::ProjectTo) = getfield(project, :info)

project_type(p::ProjectTo{T}) where {T} = T

function Base.show(io::IO, project::ProjectTo{T}) where {T}
    print(io, "ProjectTo{")
    show(io, T)
    print(io, "}")
    if isempty(backing(project))
        print(io, "()")
    else
        show(io, backing(project))
    end
end

# Structs
# Generic method is to recursively make `ProjectTo`s for all their fields. Not actually
# used on unknown structs, but useful for handling many known ones in the same manner.
function generic_projector(x::T; kw...) where {T}
    fields_nt::NamedTuple = backing(x)
    fields_proj = map(_maybe_projector, fields_nt)
    # We can't use `T` because if we have `Foo{Matrix{E}}` it should be allowed to make a
    # `Foo{Diagaonal{E}}` etc. Official API for this? https://github.com/JuliaLang/julia/issues/35543
    wrapT = T.name.wrapper
    return ProjectTo{wrapT}(; fields_proj..., kw...)
end

function generic_projection(project::ProjectTo{T}, dx::T) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(dx)
    return construct(T, map(_maybe_call, sub_projects, sub_dxs))
end

# Used for encoding fields, leaves alone non-diff types:
_maybe_projector(x::Union{AbstractArray,Number,Ref}) = ProjectTo(x)
_maybe_projector(x) = x
# Used for re-constructing fields, restores non-diff types:
_maybe_call(f::ProjectTo, x) = f(x)
_maybe_call(f, x) = f

"""
    ProjectTo(x)

Returns a `ProjectTo{T}` functor which projects a tangent `dx` onto the
relevant tangent space for `x`.

Custom `ProjectTo` methods are provided for many subtypes of `Number` (to e.g. ensure precision),
and `AbstractArray` (to e.g. ensure sparsity structure is maintained by tangent).
Called on unknown types it will (as of v1.5.0) simply return `identity`, thus can be safely
applied to arbitrary `rrule` arguments.

# Examples
```jldoctest
julia> pr = ProjectTo(1.5f0)  # preserves real numbers, and floating point precision
ProjectTo{Float32}()

julia> pr(3 + 4im)
3.0f0

julia> pd = ProjectTo(Diagonal([1,2,3]))  # preserves structured matrices
ProjectTo{Diagonal}(diag = ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(3),)),)

julia> th = @thunk reshape(1:9,3,3);

julia> pd(th) isa Thunk
true

julia> unthunk(pd(th))
3×3 Diagonal{Float64, Vector{Float64}}:
 1.0   ⋅    ⋅
  ⋅   5.0   ⋅
  ⋅    ⋅   9.0

julia> ProjectTo([1 2; 3 4]')  # no special structure, integers are promoted to float(x)
ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(2), Base.OneTo(2)))
```
"""
ProjectTo(::Any) = identity

# Generic
(::ProjectTo{T})(dx::AbstractZero) where {T} = dx
(::ProjectTo{T})(dx::NotImplemented) where {T} = dx

# Thunks
(project::ProjectTo)(dx::Thunk) = Thunk(project ∘ dx.f)
(project::ProjectTo)(dx::InplaceableThunk) = project(dx.val)

# Zero
ProjectTo(::AbstractZero) = ProjectTo{NoTangent}()  # Any x::Zero in forward pass makes this one projector,
(::ProjectTo{NoTangent})(dx) = NoTangent()          # but this is the projection only for nonzero gradients,
(::ProjectTo{NoTangent})(dx::AbstractZero) = dx     # and this one solves an ambiguity.

# Also, any explicit construction with fields, where all fields project to zero, itself
# projects to zero. This simplifies projectors for wrapper types like Diagonal([true, false]).
const _PZ = ProjectTo{<:AbstractZero}
const _PZ_Tuple = Tuple{_PZ,Vararg{_PZ}}  # 1 or more ProjectTo{<:AbstractZeros}
function ProjectTo{P}(::NamedTuple{T,<:_PZ_Tuple}) where {P,T}
    return ProjectTo{NoTangent}()
end

# Tangent
# We haven't entirely figured out when to convert Tangents to "natural" representations such as
# dx::AbstractArray (when both are possible), or the reverse. So for now we just pass them through:
(::ProjectTo{T})(dx::Tangent{<:T}) where {T} = dx

#####
##### `Base`
#####

# Bool
ProjectTo(::Bool) = ProjectTo{NoTangent}()  # same projector as ProjectTo(::AbstractZero) above

# Other never-differentiable types
for T in (:Symbol, :Char, :AbstractString, :RoundingMode, :IndexStyle)
    @eval ProjectTo(::$T) = ProjectTo{NoTangent}()
end

# Numbers
ProjectTo(::Real) = ProjectTo{Real}()
ProjectTo(::Complex) = ProjectTo{Complex}()
ProjectTo(::Number) = ProjectTo{Number}()

ProjectTo(x::Integer) = ProjectTo(float(x))
ProjectTo(x::Complex{<:Integer}) = ProjectTo(float(x))

# Preserve low-precision floats as accidental promotion is a common performance bug
for T in (Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
    @eval ProjectTo(::$T) = ProjectTo{$T}()
end

# In these cases we can just `convert` as we know we are dealing with plain and simple types
(::ProjectTo{T})(dx::AbstractFloat) where {T<:AbstractFloat} = convert(T, dx)
(::ProjectTo{T})(dx::Integer) where {T<:AbstractFloat} = convert(T, dx)  #needed to avoid ambiguity
# simple Complex{<:AbstractFloat}} cases
function (::ProjectTo{T})(dx::Complex{<:AbstractFloat}) where {T<:Complex{<:AbstractFloat}}
    return convert(T, dx)
end
(::ProjectTo{T})(dx::AbstractFloat) where {T<:Complex{<:AbstractFloat}} = convert(T, dx)
function (::ProjectTo{T})(dx::Complex{<:Integer}) where {T<:Complex{<:AbstractFloat}}
    return convert(T, dx)
end
(::ProjectTo{T})(dx::Integer) where {T<:Complex{<:AbstractFloat}} = convert(T, dx)

# Other numbers, including e.g. ForwardDiff.Dual and Symbolics.Sym, should pass through.
# We assume (lacking evidence to the contrary) that it is the right subspace of numebers.
(::ProjectTo{<:Number})(dx::Number) = dx

(project::ProjectTo{<:Real})(dx::Complex) = project(real(dx))
(project::ProjectTo{<:Complex})(dx::Real) = project(complex(dx))

# Tangents: we prefer to reconstruct numbers, but only safe to try when their constructor
# understands, including a mix of Zeros & reals. Other cases, we just let through:
(project::ProjectTo{<:Number})(dx::Tangent{<:Complex}) = project(Complex(dx.re, dx.im))
(::ProjectTo{<:Number})(dx::Tangent{<:Number}) = dx

# Arrays
# If we don't have a more specialized `ProjectTo` rule, we just assume that there is
# no structure worth re-imposing. Then any array is acceptable as a gradient.

# For arrays of numbers, just store one projector:
function ProjectTo(x::AbstractArray{T}) where {T<:Number}
    return ProjectTo{AbstractArray}(; element=_eltype_projectto(T), axes=axes(x))
end
ProjectTo(x::AbstractArray{Bool}) = ProjectTo{NoTangent}()

_eltype_projectto(::Type{T}) where {T<:Number} = ProjectTo(zero(T))
_eltype_projectto(::Type{<:Irrational}) = ProjectTo{Real}()

# In other cases, store a projector per element:
function ProjectTo(xs::AbstractArray)
    elements = map(ProjectTo, xs)
    if elements isa AbstractArray{<:ProjectTo{<:AbstractZero}}
        return ProjectTo{NoTangent}()  # short-circuit if all elements project to zero
    else
        # Arrays of arrays come here, and will apply projectors individually:
        return ProjectTo{AbstractArray}(; elements=elements, axes=axes(xs))
    end
end

function (project::ProjectTo{AbstractArray})(dx::AbstractArray{S,M}) where {S,M}
    # First deal with shape. The rule is that we reshape to add or remove trivial dimensions
    # like dx = ones(4,1), where x = ones(4), but throw an error on dx = ones(1,4) etc.
    dy = if axes(dx) === project.axes
        dx
    else
        for d in 1:max(M, length(project.axes))
            if size(dx, d) != length(get(project.axes, d, 1))
                throw(_projection_mismatch(project.axes, size(dx)))
            end
        end
        reshape(dx, project.axes)
    end
    # Then deal with the elements. One projector if AbstractArray{<:Number},
    # or one per element for arrays of anything else, including arrays of arrays:
    dz = if hasproperty(project, :element)
        T = project_type(project.element)
        S <: T ? dy : map(project.element, dy)
    else
        map((f, y) -> f(y), project.elements, dy)
    end
    return dz
end

# Trivial case, this won't collapse Any[NoTangent(), NoTangent()] but that's OK.
(project::ProjectTo{AbstractArray})(dx::AbstractArray{<:AbstractZero}) = NoTangent()

# Row vectors aren't acceptable as gradients for 1-row matrices:
function (project::ProjectTo{AbstractArray})(dx::LinearAlgebra.AdjOrTransAbsVec)
    return project(reshape(vec(dx), 1, :))
end

# Zero-dimensional arrays -- these have a habit of going missing,
# although really Ref() is probably a better structure.
function (project::ProjectTo{AbstractArray})(dx::Number) # ... so we restore from numbers
    if !(project.axes isa Tuple{})
        throw(
            DimensionMismatch(
                "array with ndims(x) == $(length(project.axes)) >  0 cannot have dx::Number"
            ),
        )
    end
    return fill(project.element(dx))
end

function _projection_mismatch(axes_x::Tuple, size_dx::Tuple)
    size_x = map(length, axes_x)
    return DimensionMismatch(
        "variable with size(x) == $size_x cannot have a gradient with size(dx) == $size_dx"
    )
end

#####
##### `Base`, part II: return of the Tangent
#####

# Ref
# Note that Ref is mutable. This causes Zygote to represent its structral tangent not as a NamedTuple,
# but as `Ref{Any}((x=val,))`. Here we use a Tangent, there is at present no mutable version, but see
# https://github.com/JuliaDiff/ChainRulesCore.jl/issues/105
function ProjectTo(x::Ref)
    sub = ProjectTo(x[])  # should we worry about isdefined(Ref{Vector{Int}}(), :x)? 
    return ProjectTo{Tangent{typeof(x)}}(; x=sub)
end
(project::ProjectTo{<:Tangent{<:Ref}})(dx::Tangent) = project(Ref(first(backing(dx))))
function (project::ProjectTo{<:Tangent{<:Ref}})(dx::Ref)
    dy = project.x(dx[])
    if dy isa AbstractZero
        return NoTangent()
    else
        return project_type(project)(; x=dy)
    end
end
# Since this works like a zero-array in broadcasting, it should also accept a number:
(project::ProjectTo{<:Tangent{<:Ref}})(dx::Number) = project(Ref(dx))

# Tuple and NamedTuple
function ProjectTo(x::Tuple)
    elements = map(ProjectTo, x)
    if elements isa NTuple{<:Any,ProjectTo{<:AbstractZero}}
        return ProjectTo{NoTangent}()
    else
        return ProjectTo{Tangent{typeof(x)}}(; elements=elements)
    end
end
function ProjectTo(x::NamedTuple)
    elements = map(ProjectTo, x)
    if Tuple(elements) isa NTuple{<:Any,ProjectTo{<:AbstractZero}}
        return ProjectTo{NoTangent}()
    else
        return ProjectTo{Tangent{typeof(x)}}(; elements...)
    end
end

# This method means that projection is re-applied to the contents of a Tangent.
# We're not entirely sure whether this is every necessary; but it should be safe,
# and should often compile away:
function (project::ProjectTo{<:Tangent{<:Union{Tuple,NamedTuple}}})(dx::Tangent)
    return project(backing(dx))
end

function (project::ProjectTo{<:Tangent{<:Tuple}})(dx::Tuple)
    len = length(project.elements)
    if length(dx) != len
        str = "tuple with length(x) == $len cannot have a gradient with length(dx) == $(length(dx))"
        throw(DimensionMismatch(str))
    end
    # Here map will fail if the lengths don't match, but gives a much less helpful error:
    dy = map((f, x) -> f(x), project.elements, dx)
    if all(d -> d isa AbstractZero, dy)
        return NoTangent()
    else
        return project_type(project)(dy...)
    end
end
function (project::ProjectTo{<:Tangent{<:NamedTuple}})(dx::NamedTuple)
    dy = _project_namedtuple(backing(project), dx)
    return project_type(project)(; dy...)
end

# Diffractor returns not necessarily a named tuple with all keys and of the same order as
# the projector
# Thus we can't use `map`
function _project_namedtuple(f::NamedTuple{fn,ft}, x::NamedTuple{xn,xt}) where {fn,ft,xn,xt}
    if @generated
        vals = Any[
            if xn[i] in fn
                :(getfield(f, $(QuoteNode(xn[i])))(getfield(x, $(QuoteNode(xn[i])))))
            else
                throw(
                    ArgumentError(
                        "named tuple with keys(x) == $fn cannot have a gradient with key $(xn[i])",
                    ),
                )
            end for i in 1:length(xn)
        ]
        :(NamedTuple{$xn}(($(vals...),)))
    else
        vals = ntuple(Val(length(xn))) do i
            name = xn[i]
            if name in fn
                getfield(f, name)(getfield(x, name))
            else
                throw(
                    ArgumentError(
                        "named tuple with keys(x) == $fn cannot have a gradient with key $(xn[i])",
                    ),
                )
            end
        end
        NamedTuple{xn}(vals)
    end
end

function (project::ProjectTo{<:Tangent{<:Tuple}})(dx::AbstractArray)
    for d in 1:ndims(dx)
        if size(dx, d) != get(length(project.elements), d, 1)
            throw(_projection_mismatch(axes(project.elements), size(dx)))
        end
    end
    dy = reshape(dx, axes(project.elements))  # allows for dx::OffsetArray
    dz = ntuple(i -> project.elements[i](dy[i]), length(project.elements))
    if all(d -> d isa AbstractZero, dy)
        return NoTangent()
    else
        return project_type(project)(dz...)
    end
end


#####
##### `LinearAlgebra`
#####

using LinearAlgebra: AdjointAbsVec, TransposeAbsVec, AdjOrTransAbsVec

# UniformScaling can represent its own cotangent
ProjectTo(x::UniformScaling) = ProjectTo{UniformScaling}(; λ=ProjectTo(x.λ))
ProjectTo(x::UniformScaling{Bool}) = ProjectTo(false)
(pr::ProjectTo{UniformScaling})(dx::UniformScaling) = UniformScaling(pr.λ(dx.λ))
(pr::ProjectTo{UniformScaling})(dx::Tangent{<:UniformScaling}) = UniformScaling(pr.λ(dx.λ))

# Row vectors
ProjectTo(x::AdjointAbsVec) = ProjectTo{Adjoint}(; parent=ProjectTo(parent(x)))
# Note that while [1 2; 3 4]' isa Adjoint, we use ProjectTo{Adjoint} only to encode AdjointAbsVec.
# Transposed matrices are, like PermutedDimsArray, just a storage detail,
# but row vectors behave differently, for example [1,2,3]' * [1,2,3] isa Number
function (project::ProjectTo{Adjoint})(dx::LinearAlgebra.AdjOrTransAbsVec)
    return adjoint(project.parent(adjoint(dx)))
end
function (project::ProjectTo{Adjoint})(dx::AbstractArray)
    if size(dx, 1) != 1 || size(dx, 2) != length(project.parent.axes[1])
        throw(_projection_mismatch((1:1, project.parent.axes...), size(dx)))
    end
    dy = eltype(dx) <: Real ? vec(dx) : adjoint(dx)
    return adjoint(project.parent(dy))
end

function ProjectTo(x::LinearAlgebra.TransposeAbsVec)
    return ProjectTo{Transpose}(; parent=ProjectTo(parent(x)))
end
function (project::ProjectTo{Transpose})(dx::LinearAlgebra.AdjOrTransAbsVec)
    return transpose(project.parent(transpose(dx)))
end
function (project::ProjectTo{Transpose})(dx::AbstractArray)
    if size(dx, 1) != 1 || size(dx, 2) != length(project.parent.axes[1])
        throw(_projection_mismatch((1:1, project.parent.axes...), size(dx)))
    end
    dy = eltype(dx) <: Number ? vec(dx) : transpose(dx)
    return transpose(project.parent(dy))
end

# Diagonal
ProjectTo(x::Diagonal) = ProjectTo{Diagonal}(; diag=ProjectTo(x.diag))
(project::ProjectTo{Diagonal})(dx::AbstractMatrix) = Diagonal(project.diag(diag(dx)))
(project::ProjectTo{Diagonal})(dx::Diagonal) = Diagonal(project.diag(dx.diag))

# Symmetric
for (SymHerm, chk, fun) in
    ((:Symmetric, :issymmetric, :transpose), (:Hermitian, :ishermitian, :adjoint))
    @eval begin
        function ProjectTo(x::$SymHerm)
            sub = ProjectTo(parent(x))
            # Because the projector stores uplo, ProjectTo(Symmetric(rand(3,3) .> 0)) isn't automatically trivial:
            sub isa ProjectTo{<:AbstractZero} && return sub
            return ProjectTo{$SymHerm}(; uplo=LinearAlgebra.sym_uplo(x.uplo), parent=sub)
        end
        function (project::ProjectTo{$SymHerm})(dx::AbstractArray)
            dy = project.parent(dx)
            # Here $chk means this is efficient on same-type.
            # If we could mutate dx, then that could speed up action on dx::Matrix.
            dz = $chk(dy) ? dy : (dy .+ $fun(dy)) ./ 2
            return $SymHerm(project.parent(dz), project.uplo)
        end
        # This is an example of a subspace which is not a subtype,
        # not clear how broadly it's worthwhile to try to support this.
        function (project::ProjectTo{$SymHerm})(dx::Diagonal)
            sub = project.parent # this is going to be unhappy about the size
            sub_one = ProjectTo{project_type(sub)}(;
                element=sub.element, axes=(sub.axes[1],)
            )
            return Diagonal(sub_one(dx.diag))
        end
    end
end

# Triangular
for UL in (:UpperTriangular, :LowerTriangular, :UnitUpperTriangular, :UnitLowerTriangular) # UpperHessenberg
    @eval begin
        ProjectTo(x::$UL) = ProjectTo{$UL}(; parent=ProjectTo(parent(x)))
        (project::ProjectTo{$UL})(dx::AbstractArray) = $UL(project.parent(dx))
        function (project::ProjectTo{$UL})(dx::Diagonal)
            sub = project.parent
            sub_one = ProjectTo{project_type(sub)}(;
                element=sub.element, axes=(sub.axes[1],)
            )
            return Diagonal(sub_one(dx.diag))
        end
    end
end

# Weird -- not exhaustive!
# one strategy is to recurse into the struct:
ProjectTo(x::Bidiagonal{T}) where {T<:Number} = generic_projector(x)
function (project::ProjectTo{Bidiagonal})(dx::AbstractMatrix)
    uplo = LinearAlgebra.sym_uplo(project.uplo)
    dv = project.dv(diag(dx))
    ev = project.ev(uplo === :U ? diag(dx, 1) : diag(dx, -1))
    return Bidiagonal(dv, ev, uplo)
end
function (project::ProjectTo{Bidiagonal})(dx::Bidiagonal)
    if project.uplo == dx.uplo
        return generic_projection(project, dx) # fast path
    else
        uplo = LinearAlgebra.sym_uplo(project.uplo)
        dv = project.dv(diag(dx))
        ev = fill!(similar(dv, length(dv) - 1), 0)
        return Bidiagonal(dv, ev, uplo)
    end
end

ProjectTo(x::SymTridiagonal{T}) where {T<:Number} = generic_projector(x)
function (project::ProjectTo{SymTridiagonal})(dx::AbstractMatrix)
    dv = project.dv(diag(dx))
    ev = project.ev((diag(dx, 1) .+ diag(dx, -1)) ./ 2)
    return SymTridiagonal(dv, ev)
end
(project::ProjectTo{SymTridiagonal})(dx::SymTridiagonal) = generic_projection(project, dx)

# another strategy is just to use the AbstractArray method
function ProjectTo(x::Tridiagonal{T}) where {T<:Number}
    notparent = invoke(ProjectTo, Tuple{AbstractArray{T2}} where {T2<:Number}, x)
    return ProjectTo{Tridiagonal}(; notparent=notparent)
end
function (project::ProjectTo{Tridiagonal})(dx::AbstractArray)
    dy = project.notparent(dx)
    return Tridiagonal(dy)
end
# Note that backing(::Tridiagonal) doesn't work, https://github.com/JuliaDiff/ChainRulesCore.jl/issues/392
