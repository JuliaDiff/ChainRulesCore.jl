"""
    AbstractZero <: AbstractTangent

Supertype for zero-like tangents—i.e., tangents that act like zero when
added or multiplied to other values.
If an AD system encounters a propagator that takes as input only subtypes of `AbstractZero`,
then it can stop performing AD operations.
All propagators are linear functions, and thus the final result will be zero.

All `AbstractZero` subtypes are singleton types.
There are two of them: [`ZeroTangent()`](@ref) and [`NoTangent()`](@ref).
"""
abstract type AbstractZero <: AbstractTangent end
Base.iszero(::AbstractZero) = true

Base.iterate(x::AbstractZero) = (x, nothing)
Base.iterate(::AbstractZero, ::Any) = nothing

Base.first(x::AbstractZero) = x
Base.tail(x::AbstractZero) = x
Base.last(x::AbstractZero) = x

Base.Broadcast.broadcastable(x::AbstractZero) = Ref(x)
Base.Broadcast.broadcasted(::Type{T}) where {T<:AbstractZero} = T()

# Linear operators
Base.adjoint(z::AbstractZero) = z
Base.transpose(z::AbstractZero) = z
Base.:/(z::AbstractZero, ::Any) = z

Base.convert(::Type{T}, x::AbstractZero) where {T<:Number} = zero(T)
# (::Type{T})(::AbstractZero, ::AbstractZero...) where {T<:Number} = zero(T)

(::Type{Complex})(x::AbstractZero, y::Real) = Complex(false, y)
(::Type{Complex})(x::Real, y::AbstractZero) = Complex(x, false)

Base.getindex(z::AbstractZero, args...) = z

Base.view(z::AbstractZero, ind...) = z
Base.sum(z::AbstractZero; dims=:) = z
Base.reshape(z::AbstractZero, size...) = z
Base.reverse(z::AbstractZero, args...; kwargs...) = z

(::Type{<:UniformScaling})(z::AbstractZero) = z

"""
    ZeroTangent() <: AbstractZero

The additive identity for tangents.
This is basically the same as `0`.
A derivative of `ZeroTangent()` does not propagate through the primal function.
"""
struct ZeroTangent <: AbstractZero end

Base.eltype(::Type{ZeroTangent}) = ZeroTangent

Base.zero(::AbstractTangent) = ZeroTangent()
Base.zero(::Type{<:AbstractTangent}) = ZeroTangent()

"""
    NoTangent() <: AbstractZero

This tangent indicates that the derivative does not exist.
It is the tangent type for primal types that are not differentiable,
such as integers or booleans (when they are not being used to represent
floating-point values).
The only valid way to perturb such values is to not change them at all.
As a consequence, `NoTangent` is functionally identical to `ZeroTangent()`,
but it provides additional semantic information.

Adding `NoTangent()` to a primal is generally wrong: gradient-based
methods cannot be used to optimize over discrete variables.
An optimization package making use of this might want to check for such a case.

!!! note
    This does not indicate that the derivative is not implemented,
    but rather that mathematically it is not defined.

This mostly shows up as the derivative with respect to dimension, index, or size
arguments.
```
    function rrule(fill, x, len::Int)
        y = fill(x, len)
        fill_pullback(ȳ) = (NoTangent(), @thunk(sum(Ȳ)), NoTangent())
        return y, fill_pullback
    end
```
"""
struct NoTangent <: AbstractZero end
