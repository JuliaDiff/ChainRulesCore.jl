using LinearAlgebra: Diagonal, diag

"""
    projector([T::Type], x)

Returns a `project(dx)` closure which maps `dx` onto type `T`, such that it is the
same size as `x`. If `T` is not provided, it is assumed to be the type of `x`.

It's necessary to have `x` to ensure that it's possible to project e.g. `AbstractZero`s
onto `Array`s -- this wouldn't be possible with type information alone because the neither
`AbstractZero`s nor `T` know what size of `Array` to produce.
"""
function projector end

projector(x) = projector(typeof(x), x)

# fallback (structs)
function projector(::Type{T}, x::T) where T
    println("to Any, T=$T")
    project(dx::T) = dx
    project(dx::AbstractZero) = zero(x)
    project(dx::AbstractThunk) = project(unthunk(dx))
    return project
end

# Numbers
function projector(::Type{T}, x::T) where {T<:Real}
    println("to Real")
    project(dx::Real) = T(dx)
    project(dx::Number) = T(real(dx)) # to avoid InexactError
    project(dx::AbstractZero) = zero(x)
    project(dx::AbstractThunk) = project(unthunk(dx))
    return project
end
function projector(::Type{T}, x::T) where {T<:Number}
    println("to Number")
    project(dx::Number) = T(dx)
    project(dx::AbstractZero) = zero(x)
    project(dx::AbstractThunk) = project(unthunk(dx))
    return project
end

# Arrays
function projector(::Type{Array{T, N}}, x::Array{T, N}) where {T, N}
    println("to Array")
    element = zero(eltype(x))
    sizex = size(x)
    project(dx::Array{T, N}) = dx # identity
    project(dx::AbstractArray) = project(collect(dx)) # from Diagonal
    project(dx::Array) = projector(element).(dx) # from different element type
    project(dx::AbstractZero) = zeros(T, sizex...)
    project(dx::AbstractThunk) = project(unthunk(dx))
    return project
end

# Tangent
function projector(::Type{<:Tangent}, x::T) where {T}
    println("to Tangent")
    project(dx) = Tangent{T}(; ((k, getproperty(dx, k)) for k in fieldnames(T))...)
    return project
end

# Diagonal
function projector(::Type{<:Diagonal{<:Any, V}}, x::Diagonal) where {V}
    println("to Diagonal")
    projV = projector(V, diag(x))
    project(dx::AbstractMatrix) = Diagonal(projV(diag(dx)))
    project(dx::Tangent) = Diagonal(projV(dx.diag))
    project(dx::AbstractZero) = Diagonal(projV(dx))
    project(dx::AbstractThunk) = project(unthunk(dx))
    return project
end

# Symmetric
function projector(::Type{<:Symmetric{<:Any, M}}, x::Symmetric) where {M}
    println("to Symetric")
    projM = projector(M, parent(x))
    uplo = Symbol(x.uplo)
    project(dx::AbstractMatrix) = Symmetric(projM(dx), uplo)
    project(dx::Tangent) = Symmetric(projM(dx.data), uplo)
    project(dx::AbstractZero) = Symmetric(projM(dx), uplo)
    project(dx::AbstractThunk) = project(unthunk(dx))
    return project
end
