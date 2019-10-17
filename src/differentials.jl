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
##### `AbstractThunk
#####
abstract type AbstractThunk <: AbstractDifferential end

Base.Broadcast.broadcastable(x::AbstractThunk) = broadcastable(extern(x))

@inline function Base.iterate(x::AbstractThunk)
    externed = extern(x)
    element, state = iterate(externed)
    return element, (externed, state)
end

@inline function Base.iterate(::AbstractThunk, (externed, state))
    element, new_state = iterate(externed, state)
    return element, (externed, new_state)
end

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
appropriately.
Propagation rule's that return multiple derivatives are not able to do all the computing themselves.
 By `@thunk`ing the work required for each, they then compute only what is needed.

#### So why not thunk everything?
`@thunk` creates a closure over the expression, which (effectively) creates a `struct`
with a field for each variable used in the expression, and call overloaded.

Do not use `@thunk` if this would be equal or more work than actually evaluating the expression itself. Examples being:
- The expression wrapping something in a `struct`, such as `Adjoint(x)` or `Diagonal(x)`
- The expression being a constant
- The expression being itself a `thunk`
- The expression being from another `rrule` or `frule` (it would be `@thunk`ed if required by the defining rule already)
"""
struct Thunk{F} <: AbstractThunk
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

# have to define this here after `@thunk` and `Thunk` is defined
Base.conj(x::AbstractThunk) = @thunk(conj(extern(x)))


(x::Thunk)() = x.f()
@inline extern(x::Thunk) = extern(x())

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
struct InplaceableThunk{T<:Thunk, F} <: AbstractThunk
    val::T
    add!::F
end

(x::InplaceableThunk)() = x.val()
@inline extern(x::InplaceableThunk) = extern(x.val)

function Base.show(io::IO, x::InplaceableThunk)
    println(io, "InplaceableThunk($(repr(x.val)), $(repr(x.add!)))")
end

# The real reason we have this:
accumulate!(Δ, ∂::InplaceableThunk) = ∂.add!(Δ)
store!(Δ, ∂::InplaceableThunk) = ∂.add!((Δ.*=false))  # zero it, then add to it.

"""
    NO_FIELDS

Constant for the reverse-mode derivative with respect to a structure that has no fields.
The most notable use for this is for the reverse-mode derivative with respect to the
function itself, when that function is not a closure.
"""
const NO_FIELDS = DNE()

"""
    refine_differential(𝒟::Type, der)

Converts, if required, a differential object `der`
(e.g. a `Number`, `AbstractDifferential`, `Matrix`, etc.),
to another  differential that is more suited for the domain given by the type 𝒟.
Often this will behave as the identity function on `der`.
"""
function refine_differential(::Type{<:Union{<:Real, AbstractArray{<:Real}}}, w::Wirtinger)
    return wirtinger_primal(w) + wirtinger_conjugate(w)
end
refine_differential(::Any, der) = der  # most of the time leave it alone.
