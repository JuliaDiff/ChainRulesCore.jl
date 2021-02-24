using LinearAlgebra: Diagonal, diag

"""
    project(T::Type, x, dx)

"project" `dx` onto type `T` such that it is the same size as `x`.

It's necessary to have `x` to ensure that it's possible to project e.g. `AbstractZero`s
onto `Array`s -- this wouldn't be possible with type information alone because the neither
`AbstractZero`s nor `T` know what size of `Array` to produce.
"""
function project end

# Number-types
project(::Type{T}, x::T, dx::T) where {T<:Real} = dx

project(::Type{T}, x::T, dx::AbstractZero) where {T<:Real} = zero(x)

project(::Type{T}, x::T, dx::AbstractThunk) where {T<:Real} = project(x, unthunk(dx))



# Arrays
project(::Type{Array{T, N}}, x::Array{T, N}, dx::Array{T, N}) where {T<:Real, N} = dx

project(::Type{<:Array{T}}, x::Array, dx::Array) where {T} = project.(Ref(T), x, dx)

function project(::Type{T}, x::Array, dx::AbstractArray) where {T<:Array}
    return project(T, x, collect(dx))
end

function project(::Type{<:Array{T}}, x::Array, dx::AbstractZero) where {T}
    return project.(Ref(T), x, Ref(dx))
end



# Diagonal
function project(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::AbstractMatrix) where {V}
    return Diagonal(project(V, diag(x), diag(dx)))
end

function project(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::Composite) where {V}
    return Diagonal(project(V, diag(x), dx.diag))
end

function project(::Type{<:Composite}, x::Diagonal, dx::Diagonal)
    return Composite{typeof(x)}(diag=diag(dx))
end



# One use for this functionality is to make it easy to define addition between two different
# representations of the same tangent. This also makes it clear that the 
Base.:(+)(x::Composite{<:Diagonal}, y::Diagonal) = x + project(typeof(x), x, y)
