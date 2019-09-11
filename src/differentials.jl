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
`@thunk(v)` is a macro that expands into `Thunk(()->v)`.


Calling a thunk, calls the wrapped closure.
`extern`ing thunks applies recursively, it also externs the differial that the closure returns.
If you do not want that, then simply call the thunk

```
julia> t = @thunk(@thunk(3))
Thunk(var"##7#9"())

julia> extern(t)
3

julia> t()
Thunk(var"##8#10"())

julia> t()()
3
```

### When to `@thunk`?
When writing `rrule`s (and to a lesser exent `frule`s), it is important to `@thunk`
appropriately. Propagation rule's that return multiple deriviatives are able to not
do all the work of computing all of them only to have just one used.
By `@thunk`ing the work required for each, they can only be computed when needed.

#### So why not thunk everything?
`@thunk` creates a closure over the expression, which is basically a struct with a
field for each variable used in the expression (closed over), and call overloaded.
If this would be equal or more work than actually evaluating the expression then don't do
it. An example would be if the expression itself is just wrapping something in a struct.
Such as `Adjoint(x)` or `Diagonal(x)`. Or if the expression is a constant, or is
itself a `Thunk`.
If you got the expression from another `rrule` (or `frule`), you don't need to
`@thunk` it since it will have been thunked if required, by the defining rule.
"""
struct Thunk{F} <: AbstractDifferential
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

(x::Thunk)() = x.f()
@inline extern(x::Thunk) = extern(x())

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

Base.show(io::IO, x::Thunk) = println(io, "Thunk($(repr(x.f)))")

"""
    InplaceableThunk(val::Thunk, add!::Function)

A wrapper for a `Thunk`, that allows it to define an inplace `add!` function,
which is used internally in `accumulate!(Δ, ::InplaceableThunk)`.

`add!` should be defined such that: `ithunk.add!(Δ) = Δ .+= ithunk.val`
but it should do this more efficently than simply doing this directly.
(Otherwise one can just use a normal `Thunk`).

Most operations on an `InplaceableThunk` treat it just like a normal `Thunk`;
and destroy its inplacability.
"""
struct InplaceableThunk{T<:Thunk, F} <: AbstractDifferential
    val::T
    add!::F
end

(x::InplaceableThunk)() = x.val()
@inline extern(x::InplaceableThunk) = extern(x.val)

Base.Broadcast.broadcastable(x::InplaceableThunk) = broadcastable(x.val)

@inline function Base.iterate(x::InplaceableThunk, args...)
    return iterate(x.val, args...)
end

Base.conj(x::InplaceableThunk) = conj(x.val)

function Base.show(io::IO, x::InplaceableThunk)
    println(io, "InplaceableThunk($(repr(x.val)), $(repr(x.add!)))")
end

# The real reason we have this:
accumulate!(Δ, ∂::InplaceableThunk) = ∂.add!(Δ)


"""
    NO_FIELDS

Constant for the reverse-mode derivative with respect to a structure that has no fields.
The most notable use for this is for the reverse-mode derivative with respect to the
function itself, when that function is not a closure.
"""
const NO_FIELDS = DNE()

"""
    differential(𝒟::Type, der)

Converts, if required, a differential object `der`
(e.g. a `Number`, `AbstractDifferential`, `Matrix`, etc.),
to another  differential that is more suited for the domain given by the type 𝒟.
Often this will behave as the identity function on `der`.
"""
function differential(::Type{<:Union{<:Real, AbstractArray{<:Real}}}, w::Wirtinger)
    return wirtinger_primal(w) + wirtinger_conjugate(w)
end
differential(::Any, der) = der  # most of the time leave it alone.
