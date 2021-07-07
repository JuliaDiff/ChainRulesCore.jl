"""
    (p::ProjectTo{T})(dx)

Projects the differential `dx` onto a specific cotangent space.
This guaranees `p(dx)::T`, except for `dx::AbstractZero`.

Usually `T` is the "outermost" part of the type, and it stores additional 
properties in `backing(p)::NamedTuple`, such as projectors for each constituent
field, and a projector `p.element` for the element type of an array of numbers.

When called on `dx::Thunk`, the projection is inserted into the thunk.
"""
struct ProjectTo{P,D<:NamedTuple}
    info::D
end
ProjectTo{P}(info::D) where {P,D<:NamedTuple} = ProjectTo{P,D}(info)
ProjectTo{P}(; kwargs...) where {P} = ProjectTo{P}(NamedTuple(kwargs))

"""
    ProjectTo(x)

Returns a `ProjectTo{T}` functor which projects a differential `dx` onto the
relevant cotangent space for `x`.

At present this undersands only `x::AbstractArray`, `x::Number`. 
It should not be called on arguments of an `rrule` method which accepts other types.

# Examples
```jldoctest
julia> r = ProjectTo(1.5)
ProjectTo{Float64}()

julia> r(3 + 4im)
3.0

julia> a = ProjectTo([1,2,3]')
ProjectTo{Adjoint}(parent = ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(3),)),)

julia> a(ones(1,3))
1×3 adjoint(::Vector{Float64}) with eltype Float64:
 1.0  1.0  1.0

julia> d = ProjectTo(Diagonal([1,2,3]));

julia> d(reshape(1:9,3,3))
3×3 Diagonal{Float64, Vector{Float64}}:
 1.0   ⋅    ⋅ 
  ⋅   5.0   ⋅ 
  ⋅    ⋅   9.0

julia> s = ProjectTo(Symmetric(rand(3,3)));

julia> s(reshape(1:9,3,3))
3×3 Symmetric{Float64, Matrix{Float64}}:
 1.0  3.0  5.0
 3.0  5.0  7.0
 5.0  7.0  9.0

julia> s(d(reshape(1:9,3,3)))  # Diagonal ! <: Symmetric, but is a sub-vector-space
3×3 Diagonal{Float64, Vector{Float64}}:
 1.0   ⋅    ⋅ 
  ⋅   5.0   ⋅ 
  ⋅    ⋅   9.0

julia> u = ProjectTo(UpperTriangular(rand(3,3) .+ im .* rand(3,3)));

julia> t = @thunk reshape(1:9,3,3,1)
Thunk(var"#9#10"())

julia> u(t) isa Thunk
true

julia> unthunk(u(t))
3×3 UpperTriangular{ComplexF64, Matrix{ComplexF64}}:
 1.0+0.0im  4.0+0.0im  7.0+0.0im
     ⋅      5.0+0.0im  8.0+0.0im
     ⋅          ⋅      9.0+0.0im

julia> ProjectTo((rand(3) .> 0)')(ones(1,3))
ZeroTangent()

julia> ProjectTo(Diagonal(rand(3) .> 0))(Diagonal(ones(3)))
ZeroTangent()

julia> bi = ProjectTo(Bidiagonal(rand(3,3), :U))
ProjectTo{Bidiagonal}(dv = ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(3),)), ev = ProjectTo{AbstractArray}(element = ProjectTo{Float64}(), axes = (Base.OneTo(2),)), uplo = ProjectTo{AbstractZero}(value = 'U',))

julia> bi(Bidiagonal(ones(ComplexF64,3,3), :U))
3×3 Bidiagonal{Float64, Vector{Float64}}:
 1.0  1.0   ⋅ 
  ⋅   1.0  1.0
  ⋅    ⋅   1.0

julia> sp = ProjectTo(sprand(3,10,0.1))
ProjectTo{SparseMatrixCSC}(element = ProjectTo{Float64}(), axes = (Base.OneTo(3), Base.OneTo(10)), rowvals = [1, 2, 2, 3, 2], nzranges = UnitRange{Int64}[1:0, 1:1, 2:2, 3:2, 3:4, 5:4, 5:5, 6:5, 6:5, 6:5], colptr = [1, 1, 2, 3, 3, 5, 5, 6, 6, 6, 6])

julia> sp(reshape(1:30, 3, 10) .+ im)
3×10 SparseMatrixCSC{Float64, Int64} with 5 stored entries:
  ⋅   4.0   ⋅    ⋅     ⋅    ⋅     ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   8.0   ⋅   14.0   ⋅   20.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   15.0   ⋅     ⋅    ⋅    ⋅    ⋅ 
```
"""
ProjectTo(x) = throw(ArgumentError("At present `ProjectTo` undersands only `x::AbstractArray`, " *
    "`x::Number`. It should not be called on arguments of an `rrule` method which accepts other types."))

Base.getproperty(p::ProjectTo, name::Symbol) = getproperty(backing(p), name)
Base.propertynames(p::ProjectTo) = propertynames(backing(p))
backing(project::ProjectTo) = getfield(project, :info)

project_type(p::ProjectTo{T}) where {T} = T
project_type(p::typeof(identity)) = Any

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

export backing, generic_projectto # for now!

# Structs
function generic_projectto(x::T; kw...) where {T}
    # Generic fallback, recursively make `ProjectTo`s for all their fields
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
function (project::ProjectTo{T})(dx::Tangent) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(canonicalize(dx))
    maybe_call(f::ProjectTo, x) = f(x)
    maybe_call(f, x) = f
    return construct(T, map(maybe_call, sub_projects, sub_dxs))
end

# Generic
(::ProjectTo{T})(dx::T) where {T} = dx 
(::ProjectTo{T})(dx::AbstractZero) where {T} = dx

# Thunks
(project::ProjectTo)(dx::Thunk) = Thunk(project ∘ dx.f)
(project::ProjectTo)(dx::InplaceableThunk) = Thunk(project ∘ dx.val.f) # can't update in-place part, but could leave it alone?
(project::ProjectTo)(dx::AbstractThunk) = project(unthunk(dx))

# Zero
ProjectTo(::AbstractZero) = ProjectTo{AbstractZero}()
(::ProjectTo{AbstractZero})(dx) = ZeroTangent()

# Bool
ProjectTo(::Bool) = ProjectTo{AbstractZero}()

# Number
ProjectTo(::T) where {T<:Number} = ProjectTo{float(T)}()
# This is quite strict, it has the virtue of preventing accidental promotion of Float32,
# but some chance that it will prevent useful behaviour, and should become Project{Real} etc.
(::ProjectTo{T})(dx::Number) where {T<:Number} = convert(T, dx)
(::ProjectTo{T})(dx::Number) where {T<:Real} = convert(T, real(dx))

# Arrays{<:Number}
# If we don't have a more specialized `ProjectTo` rule, we just assume that there is
# no structure to preserve, and any array is acceptable as a gradient.
function ProjectTo(x::AbstractArray{T}) where {T<:Number}
    element = ProjectTo(zero(T))
    # if all our elements are going to zero, then we can short circuit and just send the whole thing
    element isa ProjectTo{<:AbstractZero} && return element
    return ProjectTo{AbstractArray}(; element=element, axes=axes(x))
end
function (project::ProjectTo{AbstractArray})(dx::AbstractArray{S,M}) where {S,M}
    T = project_type(project.element)
    dy = S <: T ? dx : broadcast(project.element, dx)
    if axes(dy) == project.axes
        return dy
    else
        # The rule here is that we reshape to add or remove trivial dimensions like dx = ones(4,1),
        # where x = ones(4), but throw an error on dx = ones(1,4) etc.
        for d in 1:max(M, length(project.axes))
            size(dy, d) == length(get(project.axes, d, 1)) || throw(DimensionMismatch("wrong shape!"))
        end
        return reshape(dy, project.axes)
    end
end

# This exists to solve a Zygote bug, that a splat of an array has tuple gradient.
# Note that it can't easily be made to work with structured matrices, is that weird?
function (project::ProjectTo{AbstractArray})(dx::Tuple)
    dy = map(project.element, dx)
    return reshape(collect(dy), project.axes)
end

# Arrays of arrays -- store projector per element
ProjectTo(xs::AbstractArray{<:AbstractArray}) = ProjectTo{AbstractArray{AbstractArray}}(; elements=map(ProjectTo, xs))
function (project::ProjectTo{AbstractArray{AbstractArray}})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        for d in 1:max(ndims(dx), length(project.axes))
            size(dx, d) == length(get(project.axes, d, 1)) || throw(DimensionMismatch("wrong shape!"))
        end
        reshape(dx, project.axes)
    end
    # This always re-constructs the outer array, it's not super-lightweight
    return map((f,x) -> f(x), project.elements, dy)
end

# Arrays of other things -- same fields as Array{<:Number}, trivial element
ProjectTo(xs::AbstractArray) = ProjectTo{AbstractArray}(; element=identity, axes=axes(x))


#####
##### `LinearAlgebra`
#####

# Row vectors -- need a bit more optimising!
function ProjectTo(x::LinearAlgebra.AdjointAbsVec{T}) where {T<:Number}
    sub = ProjectTo(parent(x))
    ProjectTo{Adjoint}(; parent=sub)
end
(project::ProjectTo{Adjoint})(dx::Adjoint) = adjoint(project.parent(parent(dx)))
(project::ProjectTo{Adjoint})(dx::Transpose) = adjoint(conj(project.parent(parent(dx)))) # might copy twice?
(project::ProjectTo{Adjoint})(dx::AbstractArray) = adjoint(conj(project.parent(vec(dx)))) # not happy!

ProjectTo(x::LinearAlgebra.TransposeAbsVec{T}) where {T<:Number} = error("not yet defined")

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
for UL in (:UpperTriangular, :LowerTriangular, :UnitUpperTriangular, :UnitLowerTriangular)
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
ProjectTo(x::Bidiagonal{T}) where {T<:Number} = generic_projectto(x)
function (project::ProjectTo{Bidiagonal})(dx::AbstractMatrix)
    uplo = LinearAlgebra.sym_uplo(project.uplo)
    dv = project.dv(diag(dx))
    ev = project.ev(uplo === :U ? diag(dx, 1) : diag(dx, -1))
    return Bidiagonal(dv, ev, uplo)
end

# another strategy is just to use the AbstratArray method
function ProjectTo(x::Tridiagonal{T}) where {T<:Number}
    notparent = invoke(ProjectTo, Tuple{AbstractArray{T}} where T<:Number, x)
    ProjectTo{Tridiagonal}(; notparent = notparent)
end
function (project::ProjectTo{Tridiagonal})(dx::AbstractArray)
    dy = project.notparent(dx)
    Tridiagonal(dy)
end

#####
##### `SparseArrays`
#####

using SparseArrays
# Word from on high is that we should regard all un-stored values of sparse arrays as structural zeros.
# Thus ProjectTo needs to store nzind, and get only those. This is extremely naiive, and can probably
# be done much more efficiently by someone who knows this stuff.

function ProjectTo(x::SparseVector{T}) where {T<:Number}
    ProjectTo{SparseVector}(; element = ProjectTo(zero(T)), nzind = x.nzind, axes = axes(x))
end
function (project::ProjectTo{SparseVector})(dx::AbstractArray)
    dy = if axes(x) == project.axes
        dx
    else
        size(dx, 1) == length(project.axes[1]) || throw(DimensionMismatch("wrong shape!"))
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
        size(dx, 1) == length(project.axes[1]) || throw(DimensionMismatch("wrong shape!"))
        size(dx, 2) == length(project.axes[2]) || throw(DimensionMismatch("wrong shape!"))
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

#####
##### Utilities
#####

export MultiProject  # for now!

"""
    MultiProject(xs...)

This exists for adding projectors to rrules. 
```
function rrule(f, x, ys...)
    y = f(x, ys...)
    back(dz) = # stuff
    proj = MultiProject(f, x, ys...)
    return y, proj∘back
end
```
"""
struct MultiProject{T}
    funs::T
    function MultiProject(xs...)
        funs = map(xs) do x
            if x isa Number || x isa AbstractArray
                ProjectTo
            else
                identity
            end
        end
        new{typeof(funs)}(funs)
    end
end
(m::MultiProject)(dxs::Tuple) = map((f,dx) -> f(dx), m.funs, dxs)
