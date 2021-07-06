"""
    (p::ProjectTo{T})(dx)

Projects the differential `dx` onto a specific cotangent space.
This guaranees `p(dx)::T`, except for `dx::AbstractZero`,
but may store additional properties as a NamedTuple. 
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
ProjectTo{Adjoint{Float64, AbstractVector{Float64}}}(parent = ProjectTo{AbstractVector{Float64}}(element = ProjectTo{Float64}(), axes = (Base.OneTo(3),)),)

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

```
"""
ProjectTo

backing(project::ProjectTo) = getfield(project, :info)
Base.getproperty(p::ProjectTo, name::Symbol) = getproperty(backing(p), name)
Base.propertynames(p::ProjectTo) = propertynames(backing(p))
Base.eltype(p::ProjectTo{T}) where {T} = T

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

# fallback (structs)
function ProjectTo(x::T) where {T}
    # Generic fallback for structs, recursively make `ProjectTo`s all their fields
    fields_nt::NamedTuple = backing(x)
    return ProjectTo{T}(map(ProjectTo, fields_nt))
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

# Non-differentiable
for T in (:Bool, :Symbol, :Char, :String, :Val)
    @eval ProjectTo(dx::$T) = ProjectTo{AbstractZero}()
end
(::ProjectTo{AbstractZero})(dx::AbstractZero) = dx
(::ProjectTo{AbstractZero})(dx) = ZeroTangent()

# Number -- presently preserves Float32, but not Int, is this ideal?
ProjectTo(::T) where {T<:Number} = ProjectTo{float(T)}()
(::ProjectTo{T})(dx::Number) where {T<:Number} = convert(T, dx)
(::ProjectTo{T})(dx::Number) where {T<:Real} = convert(T, real(dx))

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
function ProjectTo(x::AbstractArray{T,N}) where {T<:Number,N}
    sub = ProjectTo(zero(T))
    return ProjectTo{AbstractArray{eltype(sub),N}}(; element=sub, axes=axes(x))
end
function (project::ProjectTo{AbstractArray{T,N}})(dx::AbstractArray{S,M}) where {T,S,N,M}
    dy = S <: T ? dx : broadcast(project.element, dx)
    if axes(dy) == project.axes
        return dy
    else
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
    ProjectTo{Adjoint{eltype(eltype(sub)), eltype(sub)}}(; parent=sub)
end
(project::ProjectTo{<:Adjoint})(dx::Adjoint) = adjoint(project.parent(parent(dx)))
(project::ProjectTo{<:Adjoint})(dx::Transpose) = adjoint(conj(project.parent(parent(dx)))) # might copy twice?
(project::ProjectTo{<:Adjoint})(dx::AbstractArray) = adjoint(conj(project.parent(vec(dx)))) # not happy!

ProjectTo(x::LinearAlgebra.TransposeAbsVec{T}) where {T<:Number} = error("not yet defined")

# Diagonal
function ProjectTo(x::Diagonal)
    sub = ProjectTo(diag(x))
    ProjectTo{Diagonal{eltype(eltype(sub)), eltype(sub)}}(; diag=sub)
end
(project::ProjectTo{<:Diagonal})(dx::AbstractMatrix) = Diagonal(project.diag(diag(dx)))
# (project::ProjectTo{T})(dx::AbstractZero) where {T<:Diagonal} = T(project.diag(dx))

# Symmetric
for (SymHerm, chk, fun) in ((:Symmetric, :issymmetric, :transpose), (:Hermitian, :ishermitian, :adjoint))
    @eval begin
        function ProjectTo(x::$SymHerm)
            sub = ProjectTo(parent(x))
            return ProjectTo{$SymHerm{eltype(eltype(sub)), eltype(sub)}}(; uplo=LinearAlgebra.sym_uplo(x.uplo), parent=sub)
        end
        function (project::ProjectTo{<:$SymHerm})(dx::AbstractArray)
            dy = project.parent(dx)
            dz = $chk(dy) ? dy : (dy .+ $fun(dy)) ./ 2
            return $SymHerm(project.parent(dz), project.uplo)
        end
        # function (project::ProjectTo{<:$SymHerm})(dx::AbstractZero)
        #     return $SymHerm(project.parent(dx), project.uplo)
        # end
        # function (project::ProjectTo{<:$SymHerm})(dx::Tangent)
        #     return $SymHerm(project.parent(dx.data), project.uplo)
        # end
    end
end

# Triangular
for UL in (:UpperTriangular, :LowerTriangular, :UnitUpperTriangular, :UnitLowerTriangular)
    @eval begin
        function ProjectTo(x::$UL)
            sub = ProjectTo(parent(x))
            return ProjectTo{$UL{eltype(eltype(sub)), eltype(sub)}}(; parent=sub)
        end
        (project::ProjectTo{<:$UL})(dx::AbstractArray) = $UL(project.parent(dx))
        # (project::ProjectTo{<:$UL})(dx::AbstractZero) = $UL(project.parent(dx))
        # (project::ProjectTo{<:$UL})(dx::Tangent) = $UL(project.parent(dx.data))
    end
end


# # SubArray
# ProjectTo(x::T) where {T<:SubArray} = ProjectTo(copy(x))  # don't project on to a view, but onto matching copy


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

julia> z, bk = proj_rrule(*, x, y);

julia> bk(z .* 100)[2] |> unthunk
3×3 Diagonal{Float64, Vector{Float64}}:
 8.04749     ⋅       ⋅ 
  ⋅       184.577    ⋅ 
  ⋅          ⋅     32.7681
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


