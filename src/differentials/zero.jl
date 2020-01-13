"""
    Zero()
The additive identity for differentials.
This is basically the same as `0`.
"""
struct Zero <: AbstractDifferential end

extern(x::Zero) = false  # false is a strong 0. E.g. `false * NaN = 0.0`

Base.Broadcast.broadcastable(::Zero) = Ref(Zero())
Base.Broadcast.broadcasted(::Type{Zero}) = Zero()

Base.iterate(x::Zero) = (x, nothing)
Base.iterate(::Zero, ::Any) = nothing

Base.zero(::AbstractDifferential) = Zero()
