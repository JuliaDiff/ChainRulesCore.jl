using LinearAlgebra: Diagonal, diag

struct ProjectTo{P, D<:NamedTuple}
    info::D
end
ProjectTo{P}(info::D) where {P,D<:NamedTuple} = ProjectTo{P,D}(info)
ProjectTo{P}(; kwargs...) where {P} = ProjectTo{P}(NamedTuple(kwargs))

backing(project::ProjectTo) = getfield(project, :info)
Base.getproperty(p::ProjectTo, name::Symbol) = getproperty(backing(p), name)
Base.propertynames(p::ProjectTo) = propertynames(backing(p))

function Base.show(io::IO, project::ProjectTo{T}) where T
    print(io, "ProjectTo{")
    show(io, T)
    print(io, "}")
    if isempty(backing(project))
        print(io, "()")
    else
        show(io, backing(project))
    end
end


"""
    ProjectTo(x)

Returns a `ProjectTo{P,...}` functor able to project a differential `dx` onto the type `T`
for a primal `x`.
This functor encloses over what ever is needed to be able to be able to do that projection.
For example, when projecting `dx=ZeroTangent()` on an array `P=Array{T, N}`, the size of `x`
is not available from `P`, so it is stored in the functor.
"""
function ProjectTo end

"""
    (::ProjectTo{T})(dx)

Projects the differential `dx` on the onto type `T`.
`ProjectTo{T}` is a functor that knows how to perform this projection.
"""
function (::ProjectTo) end

# fallback (structs)
function ProjectTo(x::T) where {T}
    # Generic fallback for structs, recursively make `ProjectTo`s all their fields
    fields_nt::NamedTuple = backing(x)
    return ProjectTo{T}(map(ProjectTo, fields_nt))
end
function (project::ProjectTo{T})(dx::Tangent) where {T}
    sub_projects = backing(project)
    sub_dxs = backing(dx)
    _call(f, x) = f(x)
    return construct(T, map(_call, sub_projects, sub_dxs))
end

# Generic
(project::ProjectTo)(dx::AbstractThunk) = project(unthunk(dx))
(::ProjectTo{T})(dx::T) where {T}  = dx  # not always true, but we can special case for when it isn't
(::ProjectTo{T})(dx::AbstractZero) where {T} = zero(T)

# Number
ProjectTo(::T) where {T<:Number} = ProjectTo{T}()
(::ProjectTo{T})(dx::Number) where {T<:Number} = convert(T, dx)
(::ProjectTo{T})(dx::Number) where {T<:Real} = convert(T, real(dx))

# Arrays 
ProjectTo(xs::T) where {T<:Array} = ProjectTo{T}(; elements=map(ProjectTo, xs))
function (project::ProjectTo{T})(dx::Array) where {T<:Array}
    _call(f, x) = f(x)
    return T(map(_call, project.elements, dx))
end
function (project::ProjectTo{T})(dx::AbstractZero) where {T<:Array}
    return T(map(proj->proj(dx), project.elements))
end
(project::ProjectTo{<:Array})(dx::AbstractArray) = project(collect(dx))

# Arrays{<:Number}: optimized case so we don't need a projector per element
ProjectTo(x::T) where {T<:Array{<:Number}} = ProjectTo{T}(; size=size(x))
(project::ProjectTo{<:Array{T}})(dx::Array) where {T<:Number} = ProjectTo(zero(T)).(dx)
(project::ProjectTo{<:Array{T}})(dx::AbstractZero) where {T<:Number} = zeros(T, project.size)

# Diagonal
ProjectTo(x::T) where {T<:Diagonal} = ProjectTo{T}(; diag=ProjectTo(diag(x)))
(project::ProjectTo{T})(dx::AbstractMatrix) where {T<:Diagonal} = T(project.diag(diag(dx)))
(project::ProjectTo{T})(dx::AbstractZero) where {T<:Diagonal} = T(project.diag(dx))

# Symmetric
ProjectTo(x::T) where {T<:Symmetric} = ProjectTo{T}(; uplo=Symbol(x.uplo), parent=ProjectTo(parent(x)))
(project::ProjectTo{<:Symmetric})(dx::AbstractMatrix) = Symmetric(project.parent(dx), project.uplo)
(project::ProjectTo{<:Symmetric})(dx::AbstractZero) = Symmetric(project.parent(dx), project.uplo)

