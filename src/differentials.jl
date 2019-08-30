#####
##### `AbstractDifferential`
#####

"""
The subtypes of `AbstractDifferential` define a custom \"algebra\" for chain
rule evaluation that attempts to factor various features like complex derivative
support, broadcast fusion, zero-elision, etc. into nicely separated parts.

All subtypes of `AbstractDifferential` implement the following operations:

`+(a, b)`: linearly combine differential `a` and differential `b`

`*(a, b)`: multiply the differential `a` by the differential `b`

`Base.conj(x)`: complex conjugate of the differential `x`

`extern(x)`: convert `x` into an appropriate non-`AbstractDifferential` type for
use outside of `ChainContext`.

Valid arguments to these operations are `T` where `T<:AbstractDifferential`, or
where `T` has proper `+` and `*` implementations.

Additionally, all subtypes of `AbstractDifferential` support `Base.iterate` and
`Base.Broadcast.broadcastable(x)`.
"""
abstract type AbstractDifferential end

Base.:+(x::AbstractDifferential) = x

"""
    extern(x)

Return `x` converted to an appropriate non-`AbstractDifferential` type, for use
with external packages that might not handle `AbstractDifferential` types.

Note that this function may return an alias (not necessarily a copy) to data
wrapped by `x`, such that mutating `extern(x)` might mutate `x` itself.
"""
@inline extern(x) = x

@inline Base.conj(x::AbstractDifferential) = x

#####
##### `Wirtinger`
#####

"""
    Wirtinger(primal::Union{Number,AbstractDifferential},
              conjugate::Union{Number,AbstractDifferential})

Returns a `Wirtinger` instance representing the complex differential:

```
df = ∂f/∂z * dz + ∂f/∂z̄ * dz̄
```

where `primal` corresponds to `∂f/∂z * dz` and `conjugate` corresponds to `∂f/∂z̄ * dz̄`.

The two fields of the returned instance can be accessed generically via the
[`wirtinger_primal`](@ref) and [`wirtinger_conjugate`](@ref) methods.
"""
struct Wirtinger{P,C} <: AbstractDifferential
    primal::P
    conjugate::C
    function Wirtinger(primal::Union{Number,AbstractDifferential},
                       conjugate::Union{Number,AbstractDifferential})
        return new{typeof(primal),typeof(conjugate)}(primal, conjugate)
    end
end

wirtinger_primal(x::Wirtinger) = x.primal
wirtinger_primal(x) = x

wirtinger_conjugate(x::Wirtinger) = x.conjugate
wirtinger_conjugate(::Any) = Zero()

extern(x::Wirtinger) = throw(ArgumentError("`Wirtinger` cannot be converted to an external type."))

Base.Broadcast.broadcastable(w::Wirtinger) = Wirtinger(broadcastable(w.primal),
                                                       broadcastable(w.conjugate))

Base.iterate(x::Wirtinger) = (x, nothing)
Base.iterate(::Wirtinger, ::Any) = nothing

# TODO: define `conj` for` `Wirtinger`
Base.conj(x::Wirtinger) = throw(MethodError(conj, x))


#####
##### `Casted`
#####

"""
    Casted(v)

This differential wraps another differential (including a number-like type)
to indicate that it should be lazily broadcast.
"""
struct Casted{V} <: AbstractDifferential
    value::V
end

cast(x) = Casted(x)
cast(f, args...) = Casted(broadcasted(f, args...))

extern(x::Casted) = materialize(broadcasted(extern, x.value))

Base.Broadcast.broadcastable(x::Casted) = x.value

Base.iterate(x::Casted) = iterate(x.value)
Base.iterate(x::Casted, state) = iterate(x.value, state)

Base.conj(x::Casted) = cast(conj, x.value)

#####
##### `Zero`
#####

"""
    Zero()
The additive identity for differentials.
This is basically the same as `0`.
"""
struct Zero <: AbstractDifferential end

extern(x::Zero) = false  # false is a strong 0. E.g. `false * NaN = 0.0`

Base.Broadcast.broadcastable(::Zero) = Ref(Zero())

Base.iterate(x::Zero) = (x, nothing)
Base.iterate(::Zero, ::Any) = nothing


#####
##### `DNE`
#####

"""
    DNE()

This differential indicates that the derivative Does Not Exist (D.N.E).
This is not the cast that it is not implemented, but rather that it mathematically
is not defined.
"""
struct DNE <: AbstractDifferential end

function extern(x::DNE)
    throw(ArgumentError("Derivative does not exit. Cannot be converted to an external type."))
end

Base.Broadcast.broadcastable(::DNE) = Ref(DNE())

Base.iterate(x::DNE) = (x, nothing)
Base.iterate(::DNE, ::Any) = nothing

#####
##### `One`
#####

"""
     One()
The Differential which is the multiplicative identity.
Basically, this represents `1`.
"""
struct One <: AbstractDifferential end

extern(x::One) = true  # true is a strong 1.

Base.Broadcast.broadcastable(::One) = Ref(One())

Base.iterate(x::One) = (x, nothing)
Base.iterate(::One, ::Any) = nothing


#####
##### `Thunk`
#####

"""
    Thunk(()->v)
A thunk is a deferred computation.
It wraps a zero argument closure that when invoked returns a differential.
"""
struct Thunk{F} <: AbstractDifferential
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

@inline extern(x::Thunk) = x.f()

Base.Broadcast.broadcastable(x::Thunk) = broadcastable(extern(x))

@inline function Base.iterate(x::Thunk)
    externed = extern(x)
    element, state = iterate(externed)
    return element, (externed, state)
end

@inline function Base.iterate(::Thunk, (externed, state))
    element, new_state = iterate(externed, state)
    return element, (externed, new_state)
end

Base.conj(x::Thunk) = @thunk(conj(extern(x)))
