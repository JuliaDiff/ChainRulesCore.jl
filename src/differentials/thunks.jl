
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


"""
    @thunk expr

Define a [`Thunk`](@ref) wrapping the `expr`, to lazily defer its evaluation.
"""
macro thunk(body)
    # Basically `:(Thunk(() -> $(esc(body))))` but use the location where it is defined.
    # so we get useful stack traces if it errors.
    func = Expr(:->, Expr(:tuple), Expr(:block, __source__, body))
    return :(Thunk($(esc(func))))
end

"""
    unthunk(x)

On `AbstractThunk`s this removes 1 layer of thunking.
On any other type, it is the identity operation.

In contrast to `extern` this is nonrecursive.
"""
@inline unthunk(x) = x

@inline extern(x::AbstractThunk) = extern(unthunk(x))

# have to define this here after `@thunk` and `Thunk` is defined
Base.conj(x::AbstractThunk) = @thunk(conj(unthunk(x)))


(x::Thunk)() = x.f()
@inline unthunk(x::Thunk) = x()

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

unthunk(x::InplaceableThunk) = unthunk(x.val)
(x::InplaceableThunk)() = unthunk(x)

function Base.show(io::IO, x::InplaceableThunk)
    println(io, "InplaceableThunk($(repr(x.val)), $(repr(x.add!)))")
end

# The real reason we have this:
accumulate!(Δ, ∂::InplaceableThunk) = ∂.add!(Δ)
store!(Δ, ∂::InplaceableThunk) = ∂.add!((Δ.*=false))  # zero it, then add to it.
