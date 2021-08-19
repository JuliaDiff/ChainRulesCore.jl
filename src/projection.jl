"""
    (p::ProjectTo{T})(dx)

Projects the differential `dx` onto a specific tangent space.

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
    ProjectTo{P}(NamedTuple(kws))
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
    # `Foo{Diagaonal{E}}` etc. We assume it has a default constructor that has all fields
    # but if it doesn't `construct` will give a good error message.
    wrapT = T.name.wrapper
    # Official API for this? https://github.com/JuliaLang/julia/issues/35543
    return ProjectTo{wrapT}(; fields_proj..., kw...)
end

function generic_projection(project::ProjectTo{T}, dx::T) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(dx)
    return construct(T, map(_maybe_call, sub_projects, sub_dxs))
end

function (project::ProjectTo{T})(dx::Tangent) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(canonicalize(dx))
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

Returns a `ProjectTo{T}` functor which projects a differential `dx` onto the
relevant tangent space for `x`.

At present this undersands only `x::Number`, `x::AbstractArray` and `x::Ref`.
It should not be called on arguments of an `rrule` method which accepts other types.

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
ProjectTo(::Any) # just to attach docstring

# Generic
(::ProjectTo{T})(dx::T) where {T} = dx  # not always correct but we have special cases for when it isn't
(::ProjectTo{T})(dx::AbstractZero) where {T} = dx
(::ProjectTo{T})(dx::NotImplemented) where {T} = dx

# Thunks
(project::ProjectTo)(dx::Thunk) = Thunk(project ∘ dx.f)

# Zero
ProjectTo(::AbstractZero) = ProjectTo{NoTangent}()  # Any x::Zero in forward pass makes this one projector,
(::ProjectTo{NoTangent})(dx) = NoTangent()          # but this is the projection only for nonzero gradients,
(::ProjectTo{NoTangent})(::NoTangent) = NoTangent() # and this one solves an ambiguity.

#####
##### `Base`
#####

# Bool
ProjectTo(::Bool) = ProjectTo{NoTangent}()  # same projector as ProjectTo(::AbstractZero) above

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
(::ProjectTo{T})(dx::AbstractFloat) where T<:AbstractFloat = convert(T, dx)
(::ProjectTo{T})(dx::Integer) where T<:AbstractFloat = convert(T, dx)  #needed to avoid ambiguity
# simple Complex{<:AbstractFloat}} cases
(::ProjectTo{T})(dx::Complex{<:AbstractFloat}) where {T<:Complex{<:AbstractFloat}} = convert(T, dx)
(::ProjectTo{T})(dx::AbstractFloat) where {T<:Complex{<:AbstractFloat}} = convert(T, dx)
(::ProjectTo{T})(dx::Complex{<:Integer}) where {T<:Complex{<:AbstractFloat}} = convert(T, dx)
(::ProjectTo{T})(dx::Integer) where {T<:Complex{<:AbstractFloat}} = convert(T, dx)

# Other numbers, including e.g. ForwardDiff.Dual and Symbolics.Sym, should pass through.
# We assume (lacking evidence to the contrary) that it is the right subspace of numebers
# The (::ProjectTo{T})(::T) method doesn't work because we are allowing a different
# Number type that might not be a subtype of the `project_type`.
(::ProjectTo{<:Number})(dx::Number) = dx

(project::ProjectTo{<:Real})(dx::Complex) = project(real(dx))
(project::ProjectTo{<:Complex})(dx::Real) = project(complex(dx))

# Arrays
# If we don't have a more specialized `ProjectTo` rule, we just assume that there is
# no structure worth re-imposing. Then any array is acceptable as a gradient.

# For arrays of numbers, just store one projector:
function ProjectTo(x::AbstractArray{T}) where {T<:Number}
    element = T <: Irrational ? ProjectTo{Real}() : ProjectTo(zero(T))
    if element isa ProjectTo{<:AbstractZero}
        return ProjectTo{NoTangent}() # short-circuit if all elements project to zero
    else
        return ProjectTo{AbstractArray}(; element=element, axes=axes(x))
    end
end

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
    dy = if axes(dx) == project.axes
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
        throw(DimensionMismatch(
            "array with ndims(x) == $(length(project.axes)) >  0 cannot have dx::Number",
        ))
    end
    return fill(project.element(dx))
end

# Accept the Tangent corresponding to a Tuple -- Zygote's splats produce these
function (project::ProjectTo{AbstractArray})(dx::Tangent{<:Any, <:Tuple})
    dy = reshape(collect(backing(dx)), project.axes)
    return project(dy)
end

# Ref -- works like a zero-array, also allows restoration from a number:
ProjectTo(x::Ref) = ProjectTo{Ref}(; x=ProjectTo(x[]))
(project::ProjectTo{Ref})(dx::Ref) = Ref(project.x(dx[]))
(project::ProjectTo{Ref})(dx::Number) = Ref(project.x(dx))

function _projection_mismatch(axes_x::Tuple, size_dx::Tuple)
    size_x = map(length, axes_x)
    return DimensionMismatch(
        "variable with size(x) == $size_x cannot have a gradient with size(dx) == $size_dx"
    )
end

#####
##### `LinearAlgebra`
#####

# Row vectors
function ProjectTo(x::LinearAlgebra.AdjointAbsVec)
    sub = ProjectTo(parent(x))
    return ProjectTo{Adjoint}(; parent=sub)
end
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
    sub = ProjectTo(parent(x))
    return ProjectTo{Transpose}(; parent=sub)
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
function ProjectTo(x::Diagonal)
    sub = ProjectTo(x.diag)
    sub isa ProjectTo{<:AbstractZero} && return sub # TODO not necc if Diagonal(NoTangent()) worked
    return ProjectTo{Diagonal}(; diag=sub)
end
(project::ProjectTo{Diagonal})(dx::AbstractMatrix) = Diagonal(project.diag(diag(dx)))
(project::ProjectTo{Diagonal})(dx::Diagonal) = Diagonal(project.diag(dx.diag))

# Symmetric
for (SymHerm, chk, fun) in (
    (:Symmetric, :issymmetric, :transpose),
    (:Hermitian, :ishermitian, :adjoint),
    )
    @eval begin
        function ProjectTo(x::$SymHerm)
            sub = ProjectTo(parent(x))
            sub isa ProjectTo{<:AbstractZero} && return sub  # TODO not necc if Hermitian(NoTangent()) etc. worked
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
        function ProjectTo(x::$UL)
            sub = ProjectTo(parent(x))
            # TODO not nesc if UnitUpperTriangular(NoTangent()) etc. worked
            sub isa ProjectTo{<:AbstractZero} && return sub
            return ProjectTo{$UL}(; parent=sub)
        end
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

#####
##### `SparseArrays`
#####

using SparseArrays
# Word from on high is that we should regard all un-stored values of sparse arrays as
# structural zeros. Thus ProjectTo needs to store nzind, and get only those.
# This implementation very naiive, can probably be made more efficient.

function ProjectTo(x::SparseVector{T}) where {T<:Number}
    return ProjectTo{SparseVector}(;
        element=ProjectTo(zero(T)), nzind=x.nzind, axes=axes(x)
    )
end
function (project::ProjectTo{SparseVector})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        if size(dx, 1) != length(project.axes[1])
            throw(_projection_mismatch(project.axes, size(dx)))
        end
        reshape(dx, project.axes)
    end
    nzval = map(i -> project.element(dy[i]), project.nzind)
    return SparseVector(length(dx), project.nzind, nzval)
end
function (project::ProjectTo{SparseVector})(dx::SparseVector)
    if size(dx) != map(length, project.axes)
        throw(_projection_mismatch(project.axes, size(dx)))
    end
    # When sparsity pattern is unchanged, all the time is in checking this,
    # perhaps some simple hash/checksum might be good enough?
    samepattern = project.nzind == dx.nzind
    # samepattern = length(project.nzind) == length(dx.nzind)
    if eltype(dx) <: project_type(project.element) && samepattern
        return dx
    elseif samepattern
        nzval = map(project.element, dx.nzval)
        SparseVector(length(dx), dx.nzind, nzval)
    else
        nzind = project.nzind
        # Or should we intersect? Can this exploit sorting?
        # nzind = intersect(project.nzind, dx.nzind)
        nzval = map(i -> project.element(dx[i]), nzind)
        return SparseVector(length(dx), nzind, nzval)
    end
end

function ProjectTo(x::SparseMatrixCSC{T}) where {T<:Number}
    return ProjectTo{SparseMatrixCSC}(;
        element=ProjectTo(zero(T)),
        axes=axes(x),
        rowval=rowvals(x),
        nzranges=nzrange.(Ref(x), axes(x, 2)),
        colptr=x.colptr,
    )
end
# You need not really store nzranges, you can get them from colptr -- TODO
# nzrange(S::AbstractSparseMatrixCSC, col::Integer) = getcolptr(S)[col]:(getcolptr(S)[col+1]-1)
function (project::ProjectTo{SparseMatrixCSC})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        if size(dx) != (length(project.axes[1]), length(project.axes[2]))
            throw(_projection_mismatch(project.axes, size(dx)))
        end
        reshape(dx, project.axes)
    end
    nzval = Vector{project_type(project.element)}(undef, length(project.rowval))
    k = 0
    for col in project.axes[2]
        for i in project.nzranges[col]
            row = project.rowval[i]
            val = dy[row, col]
            nzval[k += 1] = project.element(val)
        end
    end
    m, n = map(length, project.axes)
    return SparseMatrixCSC(m, n, project.colptr, project.rowval, nzval)
end

function (project::ProjectTo{SparseMatrixCSC})(dx::SparseMatrixCSC)
    if size(dx) != map(length, project.axes)
        throw(_projection_mismatch(project.axes, size(dx)))
    end
    samepattern = dx.colptr == project.colptr && dx.rowval == project.rowval
    # samepattern = length(dx.colptr) == length(project.colptr) && dx.colptr[end] == project.colptr[end]
    if eltype(dx) <: project_type(project.element) && samepattern
        return dx
    elseif samepattern
        nzval = map(project.element, dx.nzval)
        m, n = size(dx)
        return SparseMatrixCSC(m, n, dx.colptr, dx.rowval, nzval)
    else
        invoke(project, Tuple{AbstractArray}, dx)
    end
end
