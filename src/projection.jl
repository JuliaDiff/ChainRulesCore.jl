"""
    ProjectTo(x)

Returns a `ProjectTo{P,...}` functor able to project a differential `dx` onto the type `P`
for a primal `x`.
This functor encloses over what ever is needed to be able to be able to do that projection.
For example, when projecting `dx=ZeroTangent()` on an array `P=Array{T, N}`, the size of `x`
is not available from `P`, so it is stored in the functor.

    (::ProjectTo{P})(dx)

Projects the differential `dx` on the onto type `P`.
"""
struct ProjectTo{P,D<:NamedTuple}
    info::D
end
ProjectTo{P}(info::D) where {P,D<:NamedTuple} = ProjectTo{P,D}(info)
ProjectTo{P}(; kwargs...) where {P} = ProjectTo{P}(NamedTuple(kwargs))

backing(project::ProjectTo) = getfield(project, :info)
Base.getproperty(p::ProjectTo, name::Symbol) = getproperty(backing(p), name)
Base.propertynames(p::ProjectTo) = propertynames(backing(p))

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
(project::ProjectTo)(dx::AbstractThunk) = project(unthunk(dx))
(::ProjectTo{T})(dx::T) where {T} = dx  # not always true, but we can special case for when it isn't
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
    return T(map(proj -> proj(dx), project.elements))
end
(project::ProjectTo{<:Array})(dx::AbstractArray) = project(collect(dx))

# Arrays{<:Number}: optimized case so we don't need a projector per element
function ProjectTo(x::T) where {E<:Number,T<:Array{E}}
    return ProjectTo{T}(; element=ProjectTo(zero(E)), size=size(x))
end
(project::ProjectTo{<:Array{T}})(dx::Array) where {T<:Number} = project.element.(dx)
function (project::ProjectTo{<:Array{T}})(dx::AbstractZero) where {T<:Number}
    return zeros(T, project.size)
end
function (project::ProjectTo{<:Array{T}})(dx::Tangent{<:SubArray}) where {T<:Number}
    return project(dx.parent)
end

# Diagonal
ProjectTo(x::T) where {T<:Diagonal} = ProjectTo{T}(; diag=ProjectTo(diag(x)))
(project::ProjectTo{T})(dx::AbstractMatrix) where {T<:Diagonal} = T(project.diag(diag(dx)))
(project::ProjectTo{T})(dx::AbstractZero) where {T<:Diagonal} = T(project.diag(dx))

# :data, :uplo fields
for SymHerm in (:Symmetric, :Hermitian)
    @eval begin
        function ProjectTo(x::T) where {T<:$SymHerm}
            return ProjectTo{T}(; uplo=Symbol(x.uplo), parent=ProjectTo(parent(x)))
        end
        function (project::ProjectTo{<:$SymHerm})(dx::AbstractMatrix)
            return $SymHerm(project.parent(dx), project.uplo)
        end
        function (project::ProjectTo{<:$SymHerm})(dx::AbstractZero)
            return $SymHerm(project.parent(dx), project.uplo)
        end
        function (project::ProjectTo{<:$SymHerm})(dx::Tangent)
            return $SymHerm(project.parent(dx.data), project.uplo)
        end
    end
end

# :data field
for UL in (:UpperTriangular, :LowerTriangular)
    @eval begin
        ProjectTo(x::T) where {T<:$UL} = ProjectTo{T}(; parent=ProjectTo(parent(x)))
        (project::ProjectTo{<:$UL})(dx::AbstractMatrix) = $UL(project.parent(dx))
        (project::ProjectTo{<:$UL})(dx::AbstractZero) = $UL(project.parent(dx))
        (project::ProjectTo{<:$UL})(dx::Tangent) = $UL(project.parent(dx.data))
    end
end

# Transpose
ProjectTo(x::T) where {T<:Transpose} = ProjectTo{T}(; parent=ProjectTo(parent(x)))
function (project::ProjectTo{<:Transpose})(dx::AbstractMatrix)
    return transpose(project.parent(transpose(dx)))
end
(project::ProjectTo{<:Transpose})(dx::AbstractZero) = transpose(project.parent(dx))

# Adjoint
ProjectTo(x::T) where {T<:Adjoint} = ProjectTo{T}(; parent=ProjectTo(parent(x)))
(project::ProjectTo{<:Adjoint})(dx::AbstractMatrix) = adjoint(project.parent(adjoint(dx)))
(project::ProjectTo{<:Adjoint})(dx::AbstractZero) = adjoint(project.parent(dx))

# PermutedDimsArray
ProjectTo(x::P) where {P<:PermutedDimsArray} = ProjectTo{P}(; parent=ProjectTo(parent(x)))
function (project::ProjectTo{<:PermutedDimsArray{T,N,perm,iperm,AA}})(
    dx::AbstractArray
) where {T,N,perm,iperm,AA}
    return PermutedDimsArray{T,N,perm,iperm,AA}(permutedims(project.parent(dx), perm))
end
function (project::ProjectTo{P})(dx::AbstractZero) where {P<:PermutedDimsArray}
    return P(project.parent(dx))
end

# SubArray
ProjectTo(x::T) where {T<:SubArray} = ProjectTo(copy(x)) # don't project on to a view, but onto matching copy
