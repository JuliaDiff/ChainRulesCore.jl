"""
    AbstractZero <: AbstractDifferential

This is zero-like differential types.
If a AD system encounter a propagator taking as input only subtypes of `AbstractZero` then
it can stop performing any AD operations, as all propagator are linear functions, and thus
the final result will be zero.

All `AbstractZero` subtypes are singleton types.
There are two of them [`Zero()`](@ref) and [`DoesNotExist()`](@ref).
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

"""
    Zero() <: AbstractZero

The additive identity for differentials.
This is basically the same as `0`.
A derivative of `Zero()`. does not propagate through the primal function.
"""
struct Zero <: AbstractZero end

extern(x::Zero) = false  # false is a strong 0. E.g. `false * NaN = 0.0`

Base.eltype(::Type{Zero}) = Zero

Base.zero(::AbstractDifferential) = Zero()
Base.zero(::Type{<:AbstractDifferential}) = Zero()

"""
    DoesNotExist() <: AbstractZero

This differential indicates that the derivative does not exist.
It is the differential for a Primal type that is not differentiable.
Such an Integer, or Boolean (when not being used as a represention of a value that normally
would be a floating point.)
The only valid way to pertube such a values is to not change it at all.
As such, `DoesNotExist` is functionally identical to `Zero()`,
but provides additional semantic information.

If you are adding this differential to a primal then something is wrong.
A optimization package making use of this might like to check for such a case.

!!! note:
    This does not indicate that the derivative it is not implemented,
    but rather that mathematically it is not defined.

This mostly shows up as the deriviative with respect to dimension, index, or size
arguments.
```
    @rrule function rrule(fill, x, len::Int)
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
