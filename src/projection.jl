using LinearAlgebra: Diagonal, diag

"""
    project([T::Type], x, dx)

"project" `dx` onto type `T` such that it is the same size as `x`. If `T` is not provided,
it is assumed to be the type of `x`.

It's necessary to have `x` to ensure that it's possible to project e.g. `AbstractZero`s
onto `Array`s -- this wouldn't be possible with type information alone because the neither
`AbstractZero`s nor `T` know what size of `Array` to produce.
"""
function project end

project(x, dx) = project(typeof(x), x, dx)

# identity
project(::Type{T}, x::T, dx::T) where T = dx

### AbstractZero
project(::Type{T}, x::T, dx::AbstractZero) where T = zero(x)

### AbstractThunk
project(::Type{T}, x::T, dx::AbstractThunk) where T = project(x, unthunk(dx))


### Number-types
project(::Type{T}, x::T, dx::T2) where {T<:Number, T2<:Number} = T(dx)
project(::Type{T}, x::T, dx::Complex) where {T<:Real} = T(real(dx))


# Arrays
project(::Type{Array{T, N}}, x::Array{T, N}, dx::Array{T, N}) where {T<:Real, N} = dx

# for project([Foo(0.0), Foo(0.0)], [ZeroTangent(), ZeroTangent()])
project(::Type{<:Array{T}}, x::Array, dx::Array) where {T} = project.(Ref(T), x, dx)

# for project(rand(2, 2), Diagonal(rand(2)))
function project(::Type{T}, x::Array, dx::AbstractArray) where {T<:Array}
    return project(T, x, collect(dx))
end

# for project([Foo(0.0), Foo(0.0)], ZeroTangent())
function project(::Type{<:Array{T}}, x::Array, dx::AbstractZero) where {T}
    return project.(Ref(T), x, Ref(dx))
end


## Diagonal
function project(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::AbstractMatrix) where {V}
    return Diagonal(project(V, diag(x), diag(dx)))
end
function project(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::Tangent) where {V}
    return Diagonal(project(V, diag(x), dx.diag))
end
function project(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::AbstractZero) where {V}
    return Diagonal(project(V, diag(x), dx))
end

function project(::Type{<:Tangent}, x::Diagonal, dx::Diagonal)
    return Tangent{typeof(x)}(diag=diag(dx))
end
