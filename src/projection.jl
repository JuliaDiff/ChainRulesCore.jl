using LinearAlgebra: Diagonal, diag

"""
    projector([T::Type], x, dx)

"projector" `dx` onto type `T` such that it is the same size as `x`. If `T` is not provided,
it is assumed to be the type of `x`.

It's necessary to have `x` to ensure that it's possible to projector e.g. `AbstractZero`s
onto `Array`s -- this wouldn't be possible with type information alone because the neither
`AbstractZero`s nor `T` know what size of `Array` to produce.
""" # TODO change docstring to reflect projecor returns a closure
function projector end

projector(x, dx) = projector(typeof(x), x, dx)

# identity
projector(::Type{T}, x::T, dx::T) where T = identity

### AbstractZero
projector(::Type{T}, x::T, dx::AbstractZero) where T = _ -> zero(x)

### AbstractThunk
projector(::Type{T}, x::T, dx::AbstractThunk) where T = projector(x, unthunk(dx))


### Number-types
projector(::Type{T}, x::T, dx::T2) where {T<:Number, T2<:Number} = dx -> T(dx)
projector(::Type{T}, x::T, dx::Complex) where {T<:Real} = dx -> T(real(dx))


# Arrays
projector(::Type{Array{T, N}}, x::Array{T, N}, dx::Array{T, N}) where {T<:Real, N} = identity

# for projector([Foo(0.0), Foo(0.0)], [ZeroTangent(), ZeroTangent()])
projector(::Type{<:Array{T}}, x::Array, dx::Array) where {T} = projector.(Ref(T), x, dx) # TODO

# for projector(rand(2, 2), Diagonal(rand(2)))
function projector(::Type{T}, x::Array, dx::AbstractArray) where {T<:Array}
    return projector(T, x, collect(dx))
end

# for projector([Foo(0.0), Foo(0.0)], ZeroTangent())
function projector(::Type{<:Array{T}}, x::Array, dx::AbstractZero) where {T}
    return projector.(Ref(T), x, Ref(dx)) # TODO
end


## Diagonal
function projector(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::AbstractMatrix) where {V}
    d = diag(x)
    return dx -> Diagonal(projector(V, d, diag(dx)))
end
function projector(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::Tangent) where {V}
    d = diag(x)
    return dx -> Diagonal(projector(V, d, dx.diag))
end
function projector(::Type{<:Diagonal{<:Any, V}}, x::Diagonal, dx::AbstractZero) where {V}
    d = diag(x)
    return dx -> Diagonal(projector(V, d, dx))
end

function projector(::Type{<:Tangent}, x::Diagonal, dx::Diagonal)
    T = typeof(x)
    return dx -> Tangent{T}(diag=diag(dx))
end
