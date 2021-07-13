"""
    (p::ProjectTo{T})(dx)

Projects the differential `dx` onto a specific cotangent space.
This guaranees `p(dx)::T`, except for allowing `dx::AbstractZero` to pass through.

Usually `T` is the "outermost" part of the type, and `p` stores additional 
properties such as projectors for each constituent field,
and a projector `p.element` for the element type of an array of numbers.
These properties can be supplied as keyword arguments on construction,
`p = ProjectTo{T}(; field=data, element=Projector(x))`. For each `T` in use,
corresponding methods should be written for `ProjectTo{T}(dx)` with nonzero `dx`.

When called on `dx::Thunk`, the projection is inserted into the thunk.
"""
struct ProjectTo{P,D<:NamedTuple}
    info::D
end
ProjectTo{P}(info::D) where {P,D<:NamedTuple} = ProjectTo{P,D}(info)
ProjectTo{P}(; kwargs...) where {P} = ProjectTo{P}(NamedTuple(kwargs))

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
    fields_proj = map(fields_nt) do x1
        if x1 isa Number || x1 isa AbstractArray
            ProjectTo(x1)
        else
            x1
        end
    end        
    # We can't use `T` because if we have `Foo{Matrix{E}}` it should be allowed to make a
    # `Foo{Diagaonal{E}}` etc. We assume it has a default constructor that has all fields 
    # but if it doesn't `construct` will give a good error message.
    wrapT = T.name.wrapper
    return ProjectTo{wrapT}(; fields_proj..., kw...)
end

function generic_projection(project::ProjectTo{T}, dx::T) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(dx)
    return construct(T, map(_maybe_project, sub_projects, sub_dxs))
end

function (project::ProjectTo{T})(dx::Tangent) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(canonicalize(dx))
    return construct(T, map(_maybe_project, sub_projects, sub_dxs))
end

_maybe_project(f::ProjectTo, x) = f(x)
_maybe_project(f, x) = f

"""
    ProjectTo(x)

Returns a `ProjectTo{T}` functor which projects a differential `dx` onto the
relevant cotangent space for `x`.

At present this undersands only `x::Number`, `x::AbstractArray` and `x::Ref`.
It should not be called on arguments of an `rrule` method which accepts other types.

# Examples
```jldoctest
julia> r = ProjectTo(1.5f0)  # preserves real numbers, and floating point precision
ProjectTo{Float32}()

julia> r(3 + 4im)
3.0f0

julia> d = ProjectTo(Diagonal([1,2,3]))  # preserves structured matrices
ProjectTo{Diagonal}(diag = ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(3),)),)

julia> t = @thunk reshape(1:9,3,3);

julia> d(t) isa Thunk
true

julia> unthunk(d(t))  # integers are promoted to float(x)
3×3 Diagonal{Float64, Vector{Float64}}:
 1.0   ⋅    ⋅ 
  ⋅   5.0   ⋅ 
  ⋅    ⋅   9.0
```
"""
ProjectTo() = ProjectTo{Any}()  # trivial case, exists so that maybe_call(f, x) knows what to do
(x::ProjectTo{Any})(dx) = dx

# Generic
(::ProjectTo{T})(dx::T) where {T} = dx  # not always correct but we have special cases for when it isn't
(::ProjectTo{T})(dx::AbstractZero) where {T} = dx
(::ProjectTo{T})(dx::NotImplemented) where {T} = dx

# Thunks
(project::ProjectTo)(dx::Thunk) = Thunk(project ∘ dx.f)
(project::ProjectTo)(dx::InplaceableThunk) = project(dx.val)  # can't update in-place part
(project::ProjectTo)(dx::AbstractThunk) = project(unthunk(dx))

# Zero
ProjectTo(::AbstractZero) = ProjectTo{AbstractZero}()
(::ProjectTo{AbstractZero})(dx) = ZeroTangent()

#####
##### `Base`
#####

# Bool
ProjectTo(::Bool) = ProjectTo{AbstractZero}()

# Numbers
ProjectTo(::Real) = ProjectTo{Real}()
ProjectTo(::Complex) = ProjectTo{Complex}()
ProjectTo(::Number) = ProjectTo{Number}()
for T in (Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
    # Preserve low-precision floats as accidental promotion is a common perforance bug
    @eval ProjectTo(::$T) = ProjectTo{$T}()
end
ProjectTo(x::Integer) = ProjectTo(float(x))
ProjectTo(x::Complex{<:Integer}) = ProjectTo(float(x))
(::ProjectTo{T})(dx::Number) where {T<:Number} = convert(T, dx)
(::ProjectTo{T})(dx::Number) where {T<:Real} = convert(T, real(dx))

# Arrays{<:Number}
# If we don't have a more specialized `ProjectTo` rule, we just assume that there is
# no structure worth preserving. Then any array is acceptable as a gradient.
function ProjectTo(x::AbstractArray{T}) where {T<:Number}
    element = ProjectTo(zero(T))
    # if all our elements are going to zero, then we can short circuit and just send the whole thing
    element isa ProjectTo{<:AbstractZero} && return element
    return ProjectTo{AbstractArray}(; element=element, axes=axes(x))
end
function (project::ProjectTo{AbstractArray})(dx::AbstractArray{S,M}) where {S,M}
    T = project_type(project.element)
    dy = S <: T ? dx : map(project.element, dx)
    if axes(dy) == project.axes
        return dy
    else
        # The rule here is that we reshape to add or remove trivial dimensions like dx = ones(4,1),
        # where x = ones(4), but throw an error on dx = ones(1,4) etc.
        for d in 1:max(M, length(project.axes))
            size(dy, d) == length(get(project.axes, d, 1)) || throw(_projection_mismatch(project.axes, size(dy)))
        end
        return reshape(dy, project.axes)
    end
end

# Zero-dimensional arrays -- these have a habit of going missing:
function (project::ProjectTo{AbstractArray})(dx::Number) # ... so we restore from numbers
    project.axes isa Tuple{} || sum(length, project.axes) == 1 || throw(_projection_mismatch(project.axes, size(dx)))
    fill(project.element(dx))
end

# Arrays of arrays -- store projector per element:
ProjectTo(xs::AbstractArray{<:AbstractArray}) = ProjectTo{AbstractArray{AbstractArray}}(; elements=map(ProjectTo, xs), axes = axes(xs))
function (project::ProjectTo{AbstractArray{AbstractArray}})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        for d in 1:max(ndims(dx), length(project.axes))
            size(dx, d) == length(get(project.axes, d, 1)) || throw(_projection_mismatch(project.axes, size(dx)))
        end
        reshape(dx, project.axes)
    end
    # This always re-constructs the outer array, it's not super-lightweight
    return map((f,x) -> f(x), project.elements, dy)
end

# Arrays of other things -- since we've said we support arrays, but may not support their elements,
# we handle the container as above but store trivial element projector:
ProjectTo(xs::AbstractArray) = ProjectTo{AbstractArray}(; element=ProjectTo(), axes=axes(xs))

# Ref -- likewise aim at containers of supported things, but treat unsupported trivially.
ProjectTo(x::Ref{<:Number}) = ProjectTo{Ref}(; x = ProjectTo(getindex(x)))
ProjectTo(x::Ref{<:AbstractArray}) = ProjectTo{Ref}(; x = ProjectTo(getindex(x)))
ProjectTo(x::Ref) = ProjectTo{Ref}(; x = ProjectTo())
(project::ProjectTo{Ref})(dx::Ref) = Ref(project.x(dx[]))
# And like zero-dim arrays, allow restoration from a number:
(project::ProjectTo{Ref})(dx::Number) = Ref(project.x(dx))

function _projection_mismatch(axes_x::Tuple, size_dx::Tuple)
    size_x = map(length, axes_x)
    DimensionMismatch("variable with size(x) == $size_x cannot have a gradient with size(dx) == $size_dx")
end

#####
##### `LinearAlgebra`
#####

# Row vectors
function ProjectTo(x::LinearAlgebra.AdjointAbsVec{T}) where {T<:Number}
    sub = ProjectTo(parent(x))
    ProjectTo{Adjoint}(; parent=sub)
end
# Note that while [1 2; 3 4]' isa Adjoint, we use ProjectTo{Adjoint} only to encode AdjointAbsVec.
# Transposed matrices are, like PermutedDimsArray, just a storage detail,
# but while row vectors behave differently, for example [1,2,3]' * [1,2,3] isa Number
(project::ProjectTo{Adjoint})(dx::Adjoint) = adjoint(project.parent(parent(dx)))
(project::ProjectTo{Adjoint})(dx::Transpose) = adjoint(conj(project.parent(parent(dx)))) # might copy twice?
function (project::ProjectTo{Adjoint})(dx::AbstractArray)
    size(dx,1) == 1 && size(dx,2) == length(project.parent.axes[1]) || throw(_projection_mismatch((1:1, project.parent.axes...), size(dx)))
    dy = project.parent(vec(dx))
    return adjoint(conj(dy))
end

function ProjectTo(x::LinearAlgebra.TransposeAbsVec{T}) where {T<:Number}
    sub = ProjectTo(parent(x))
    ProjectTo{Transpose}(; parent=sub)
end
(project::ProjectTo{Transpose})(dx::Transpose) = transpose(project.parent(parent(dx)))
(project::ProjectTo{Transpose})(dx::Adjoint) = transpose(conj(project.parent(parent(dx))))
function (project::ProjectTo{Transpose})(dx::AbstractArray)
    size(dx,1) == 1 && size(dx,2) == length(project.parent.axes[1]) || throw(_projection_mismatch((1:1, project.parent.axes...), size(dx)))
    dy = project.parent(vec(dx))
    return transpose(dy)
end

# Diagonal
function ProjectTo(x::Diagonal)
    eltype(x) == Bool && return ProjectTo(false)
    sub = ProjectTo(diag(x))
    return ProjectTo{Diagonal}(; diag=sub)
end
(project::ProjectTo{Diagonal})(dx::AbstractMatrix) = Diagonal(project.diag(diag(dx)))

# Symmetric
for (SymHerm, chk, fun) in ((:Symmetric, :issymmetric, :transpose), (:Hermitian, :ishermitian, :adjoint))
    @eval begin
        function ProjectTo(x::$SymHerm)
            eltype(x) == Bool && return ProjectTo(false)
            sub = ProjectTo(parent(x))
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
            sub_one = ProjectTo{project_type(sub)}(; element = sub.element, axes = (sub.axes[1],))
            return Diagonal(sub_one(dx.diag))
        end
    end
end

# Triangular
for UL in (:UpperTriangular, :LowerTriangular, :UnitUpperTriangular, :UnitLowerTriangular) # UpperHessenberg
    @eval begin
        function ProjectTo(x::$UL)
            eltype(x) == Bool && return ProjectTo(false)
            sub = ProjectTo(parent(x))
            return ProjectTo{$UL}(; parent=sub)
        end
        (project::ProjectTo{$UL})(dx::AbstractArray) = $UL(project.parent(dx))
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
(project::ProjectTo{Bidiagonal})(dx::Bidiagonal) = generic_projection(project, dx)

ProjectTo(x::SymTridiagonal{T}) where {T<:Number} = generic_projector(x)
function (project::ProjectTo{SymTridiagonal})(dx::AbstractMatrix)
    dv = project.dv(diag(dx))
    ev = project.ev((diag(dx, 1) .+ diag(dx, -1)) ./ 2)
    return SymTridiagonal(dv, ev)
end
(project::ProjectTo{SymTridiagonal})(dx::SymTridiagonal) = generic_projection(project, dx)

# another strategy is just to use the AbstratArray method
function ProjectTo(x::Tridiagonal{T}) where {T<:Number}
    notparent = invoke(ProjectTo, Tuple{AbstractArray{T}} where T<:Number, x)
    ProjectTo{Tridiagonal}(; notparent = notparent)
end
function (project::ProjectTo{Tridiagonal})(dx::AbstractArray)
    dy = project.notparent(dx)
    Tridiagonal(dy)
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
    ProjectTo{SparseVector}(; element = ProjectTo(zero(T)), nzind = x.nzind, axes = axes(x))
end
function (project::ProjectTo{SparseVector})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        size(dx, 1) == length(project.axes[1]) || throw(_projection_mismatch(project.axes, size(dx)))
        reshape(dx, project.axes)
    end
    nzval = map(i -> project.element(dy[i]), project.nzind)
    n = length(project.axes[1])
    return SparseVector(n, project.nzind, nzval)
end

function ProjectTo(x::SparseMatrixCSC{T}) where {T<:Number}
    ProjectTo{SparseMatrixCSC}(; element = ProjectTo(zero(T)), axes = axes(x),
        rowvals = rowvals(x), nzranges = nzrange.(Ref(x), axes(x,2)), colptr = x.colptr)
end
# You need not really store nzranges, you can get them from colptr
# nzrange(S::AbstractSparseMatrixCSC, col::Integer) = getcolptr(S)[col]:(getcolptr(S)[col+1]-1)
function (project::ProjectTo{SparseMatrixCSC})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        size(dx, 1) == length(project.axes[1]) || throw(_projection_mismatch(project.axes, size(dx)))
        size(dx, 2) == length(project.axes[2]) || throw(_projection_mismatch(project.axes, size(dx)))
        reshape(dx, project.axes)
    end
    nzval = Vector{project_type(project.element)}(undef, length(project.rowvals))
    k = 0
    for col in project.axes[2]
        for i in project.nzranges[col]
            row = project.rowvals[i]
            val = dy[row, col]
            nzval[k+=1] = project.element(val)
        end
    end
    m, n = length.(project.axes)
    return SparseMatrixCSC(m, n, project.colptr, project.rowvals, nzval)
end
