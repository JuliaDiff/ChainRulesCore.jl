"""
    AbstractZero <: AbstractDifferential

Supertype for zero-like differentials—i.e., differentials that act like zero when
added or multiplied to other values.
If an AD system encounters a propagator that takes as input only subtypes of `AbstractZero`,
then it can stop performing AD operations.
All propagators are linear functions, and thus the final result will be zero.

All `AbstractZero` subtypes are singleton types.
There are two of them: [`Zero()`](@ref) and [`DoesNotExist()`](@ref).
"""
abstract type AbstractZero <: AbstractDifferential end
Base.iszero(::AbstractZero) = true

Base.iterate(x::AbstractZero) = (x, nothing)
Base.iterate(::AbstractZero, ::Any) = nothing

Base.Broadcast.broadcastable(x::AbstractZero) = Ref(x)
Base.Broadcast.broadcasted(::Type{T}) where T<:AbstractZero = T()

# Linear operators
Base.adjoint(z::AbstractZero) = z
Base.transpose(z::AbstractZero) = z
Base.:/(z::AbstractZero, ::Any) = z

Base.convert(::Type{T}, x::AbstractZero) where T <: Number = zero(T)

"""
    Zero() <: AbstractZero

The additive identity for differentials.
This is basically the same as `0`.
A derivative of `Zero()` does not propagate through the primal function.
"""
struct Zero <: AbstractZero end

extern(x::Zero) = false  # false is a strong 0. E.g. `false * NaN = 0.0`

Base.eltype(::Type{Zero}) = Zero

Base.zero(::AbstractDifferential) = Zero()
Base.zero(::Type{<:AbstractDifferential}) = Zero()

"""
    DoesNotExist() <: AbstractZero

This differential indicates that the derivative does not exist.
It is the differential for primal types that are not differentiable,
such as integers or booleans (when they are not being used to represent
floating-point values).
The only valid way to perturb such values is to not change them at all.
As a consequence, `DoesNotExist` is functionally identical to `Zero()`,
but it provides additional semantic information.

Adding this differential to a primal is generally wrong: gradient-based
methods cannot be used to optimize over discrete variables.
An optimization package making use of this might want to check for such a case.

!!! note:
    This does not indicate that the derivative is not implemented,
    but rather that mathematically it is not defined.

This mostly shows up as the derivative with respect to dimension, index, or size
arguments.
```
    function rrule(fill, x, len::Int)
        y = fill(x, len)
        fill_pullback(ȳ) = (NO_FIELDS, @thunk(sum(Ȳ)), DoesNotExist())
        return y, fill_pullback
    end
```
"""
struct DoesNotExist <: AbstractZero end

function extern(x::DoesNotExist)
    throw(ArgumentError("Derivative does not exit. Cannot be converted to an external type."))
end
