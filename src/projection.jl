"""
    (p::ProjectTo{T})(dx)

Projects the differential `dx` onto a specific cotangent space.
This guaranees `p(dx)::T`, except for `dx::AbstractZero`.

In addition, it typically stores additional properties in `backing(p)::NamedTuple`,
such as projectors for each constituent field, 
and a projector for the element type of an array.

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

julia> s(d(reshape(1:9,3,3)))
3×3 Symmetric{Float64, Diagonal{Float64, Vector{Float64}}}:
 1.0  0.0  0.0
 0.0  5.0  0.0
 0.0  0.0  9.0

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
```
"""
ProjectTo(x) = generic_projectto(x)

backing(project::ProjectTo) = getfield(project, :info)
Base.getproperty(p::ProjectTo, name::Symbol) = getproperty(backing(p), name)
Base.propertynames(p::ProjectTo) = propertynames(backing(p))
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

export backing, generic_projectto # for now!

# fallback (structs)
function generic_projectto(x::T) where {T}
    # Generic fallback for structs, recursively make `ProjectTo`s all their fields
    fields_nt::NamedTuple = backing(x)
    # We can't use `T` because if we have `Foo{Matrix{E}}` it should be allowed to make a `Foo{Diagaonal{E}}` etc
    # we assume it has a default constructor that has all fields but if it doesn't `construct` will give a good error message
    wrapT = T.name.wrapper
    return ProjectTo{wrapT}(map(ProjectTo, fields_nt))
end
function (project::ProjectTo{T})(dx::Tangent) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(canonicalize(dx))
    _call(f, x) = f(x)
    return construct(T, map(_call, sub_projects, sub_dxs))
end

# should not work for Tuples and NamedTuples, as not valid tangent types
function ProjectTo(x::T) where {T<:Union{<:Tuple,NamedTuple}}
    return throw(
        ArgumentError("The `x` in `ProjectTo(x)` must be a valid differential, not $x")
    )
end

# Generic
(::ProjectTo{T})(dx::T) where {T} = dx  # not always true, but we can special case for when it isn't
(::ProjectTo{T})(dx::AbstractZero) where {T} = dx # zero(T)

# Thunks
(project::ProjectTo)(dx::Thunk) = Thunk(project ∘ dx.f)
(project::ProjectTo)(dx::InplaceableThunk) = Thunk(project ∘ dx.val.f) # can't update in-place part
(project::ProjectTo)(dx::AbstractThunk) = project(unthunk(dx))

# Non-differentiable -- not so sure this is necessary. Keeping value like this is a bit awkward.
for T in (:Bool, :Symbol, :Char, :String, :Val, :Type)
    @eval ProjectTo(dx::$T) = ProjectTo{AbstractZero}(; value=dx)
end
(::ProjectTo{AbstractZero})(dx::AbstractZero) = dx
(::ProjectTo{AbstractZero})(dx) = ZeroTangent()

# Number -- presently preserves Float32, but not Int, is this ideal?
ProjectTo(::T) where {T<:Number} = ProjectTo{float(T)}()
(::ProjectTo{T})(dx::Number) where {T<:Number} = convert(T, dx)
(::ProjectTo{T})(dx::Number) where {T<:Real} = convert(T, real(dx))

ProjectTo(::Type{T}) where T<:Number = ProjectTo(zero(T)) # maybe

# Arrays 
# ProjectTo(xs::T) where {T<:Array} = ProjectTo{T}(; elements=map(ProjectTo, xs))
# function (project::ProjectTo{T})(dx::Array) where {T<:Array}
#     _call(f, x) = f(x)
#     return T(map(_call, project.elements, dx))
# end
# function (project::ProjectTo{T})(dx::AbstractZero) where {T<:Array}
#     return T(map(proj -> proj(dx), project.elements))
# end
# (project::ProjectTo{<:Array})(dx::AbstractArray) = project(collect(dx))

# Arrays{<:Number}: optimized case so we don't need a projector per element
# If we don't have a more specialized `ProjectTo` rule, we just assume that it is some kind
# of dense array without any structure, etc that we might need to preserve
function ProjectTo(x::AbstractArray{T,N}) where {T<:Number,N}
    sub = ProjectTo(zero(T))
    # if all our elements are going to zero, then we can short circuit and just send the whole thing
    sub isa ProjectTo{<:AbstractZero} && return sub
    return ProjectTo{AbstractArray}(; element=sub, axes=axes(x))
end
function (project::ProjectTo{AbstractArray})(dx::AbstractArray{S,M}) where {S,M}
    T = project_type(project.element)
    N = length(project.axes)
    dy = S <: T ? dx : broadcast(project.element, dx)
    if axes(dy) == project.axes
        return dy
    else
        # the rule here is that we reshape to add or remove trivial dimensions like dx = ones(4,1),
        # where x = ones(4), but throw an error on dx = ones(1,4) etc.
        for d in 1:max(N,M)
            size(dy, d) == length(get(project.axes, d, 1)) || throw(DimensionMismatch("wrong shape!"))
        end
        return reshape(dy, project.axes)
    end
end
# function (project::ProjectTo{<:Array{T}})(dx::Tangent{<:SubArray}) where {T<:Number}
#     return project(dx.parent)
# end

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
    ProjectTo{Diagonal}(; diag=sub)
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

# Weird
ProjectTo(x::Bidiagonal) = generic_projectto(x) # not sure!
function (project::ProjectTo{Bidiagonal})(dx::AbstractMatrix)
    uplo = LinearAlgebra.sym_uplo(project.uplo.value)
    dv = project.dv(diag(dx))
    ev = project.ev(uplo === :U ? diag(dx, 1) : diag(dx, -1))
    Bidiagonal(dv, ev, uplo)
end

#=

x = LinearAlgebra.Tridiagonal(rand(4,4))
backing(x) # UndefRefError: access to undefined reference

=#


# # SubArray
# ProjectTo(x::T) where {T<:SubArray} = ProjectTo(copy(x))  # don't project on to a view, but onto matching copy

# Sparse

# using SparseArrays



export proj_rrule
"""
    proj_rrule(f, args...)

This calls the corresponding `rrule`, but automatically
applies ProjectTo to all arguments.

```jldoctest
julia> using ChainRules

julia> x, y = Diagonal(rand(3)), rand(3,3);

julia> z, bk = rrule(*, x, y);

julia> bk(z .* 100)[2] |> unthunk
3×3 Matrix{Float64}:
  8.04749   20.4969   16.3239
 39.2846   184.577   128.653
 11.1317    45.7744   32.7681

julia> z2, bk2 = proj_rrule(*, x, y);

julia> bk2(z .* 100)[2] |> unthunk  # now preserves the subspace
3×3 Diagonal{Float64, Vector{Float64}}:
 8.04749     ⋅       ⋅ 
  ⋅       184.577    ⋅ 
  ⋅          ⋅     32.7681

julia> @btime rrule(*, \$x, \$y)[2](\$z)[2]|>unthunk;
  66.777 ns (2 allocations: 256 bytes)

julia> @btime proj_rrule(*, \$x, \$y)[2](\$z)[2]|>unthunk;  # still not ideal, makes & throws away Matrix
  117.798 ns (4 allocations: 416 bytes)

julia> _, bk3 = rrule(*, 1, 2+3im)
(2 + 3im, ChainRules.var"#times_pullback#1114"{Int64, Complex{Int64}}(1, 2 + 3im))

julia> bk3(4+5im)
(NoTangent(), 23 - 2im, 4 + 5im)

julia> _, bk4 = proj_rrule(*, 1, 2+3im);

julia> bk4(4+5im)  # works for scalars too
(NoTangent(), 23.0, 4.0 + 5.0im)

```
"""
function proj_rrule(f, args...)
    ps = map(ProjectTo, args)
    y, back = rrule(f, args...)
    function proj_back(dy)
        res = back(dy)
        (first(res), map((p,dx) -> p(dx), ps, Base.tail(res))...)
    end
    y, proj_back
end

