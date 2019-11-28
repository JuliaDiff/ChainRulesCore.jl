"""
    DoesNotExist()

This differential indicates that the derivative Does Not Exist (D.N.E).
This is not the cast that it is not implemented, but rather that it mathematically
is not defined.
"""
struct DoesNotExist <: AbstractDifferential end

function extern(x::DoesNotExist)
    throw(ArgumentError("Derivative does not exit. Cannot be converted to an external type."))
end

Base.Broadcast.broadcastable(::DoesNotExist) = Ref(DoesNotExist())

Base.iterate(x::DoesNotExist) = (x, nothing)
Base.iterate(::DoesNotExist, ::Any) = nothing

