"""
    DoesNotExist()

This differential indicates that the derivative Does Not Exist (D.N.E).
It is the differential for a Primal type that is not differentiable.
Such an Integer, or Boolean (when not being used as a represention of a valid that normally
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
    function rrule(fill, x, len::Int)
        y = fill(x, len)
        fill_pullback(ȳ) = (NO_FIELDS, @thunk(sum(Ȳ)), DoesNotExist())
        return y, fill_pullback
    end
```
"""
struct DoesNotExist <: AbstractDifferential end

function extern(x::DoesNotExist)
    throw(ArgumentError("Derivative does not exit. Cannot be converted to an external type."))
end

Base.Broadcast.broadcastable(::DoesNotExist) = Ref(DoesNotExist())

Base.iterate(x::DoesNotExist) = (x, nothing)
Base.iterate(::DoesNotExist, ::Any) = nothing
