"""
     One()
The Differential which is the multiplicative identity.
Basically, this represents `1`.
"""
struct One <: AbstractDifferential end

extern(x::One) = true  # true is a strong 1.

Base.Broadcast.broadcastable(::One) = Ref(One())
Base.Broadcast.broadcasted(::Type{One}) = One()

Base.iterate(x::One) = (x, nothing)
Base.iterate(::One, ::Any) = nothing
