abstract type AbstractThunk <: AbstractTangent end

Base.Broadcast.broadcastable(x::AbstractThunk) = broadcastable(unthunk(x))

@inline function Base.iterate(x::AbstractThunk)
    val = unthunk(x)
    element, state = iterate(val)
    return element, (val, state)
end

@inline function Base.iterate(::AbstractThunk, (val, state))
    element, new_state = iterate(val, state)
    return element, (val, new_state)
end

"""
    @thunk expr

Define a [`ThunkedTangent`](@ref) wrapping the `expr`, to lazily defer its evaluation.
"""
macro thunk(body)
    # Basically `:(ThunkedTangent(() -> $(esc(body))))` but use the location where it is defined.
    # so we get useful stack traces if it errors.
    func = Expr(:->, Expr(:tuple), Expr(:block, __source__, body))
    return :(ThunkedTangent($(esc(func))))
end

"""
    unthunk(x)

On `AbstractThunk`s this removes 1 layer of thunking.
On any other type, it is the identity operation.

In contrast to [`extern`](@ref) this is nonrecursive.
"""
@inline unthunk(x) = x

@inline extern(x::AbstractThunk) = extern(unthunk(x))

Base.conj(x::AbstractThunk) = @thunk(conj(unthunk(x)))
Base.adjoint(x::AbstractThunk) = @thunk(adjoint(unthunk(x)))
Base.transpose(x::AbstractThunk) = @thunk(transpose(unthunk(x)))

#####
##### `ThunkedTangent`
#####

"""
    ThunkedTangent(()->v)
A thunk is a deferred computation.
It wraps a zero argument closure that when invoked returns a differential.
`@thunk(v)` is a macro that expands into `ThunkedTangent(()->v)`.

Calling a thunk, calls the wrapped closure.
If you are unsure if you have a `ThunkedTangent`, call [`unthunk`](@ref) which is a no-op when the
argument is not a `ThunkedTangent`.
If you need to unthunk recursively, call [`extern`](@ref), which also externs the differial
that the closure returns.

```jldoctest
julia> t = @thunk(@thunk(3))
ThunkedTangent(var"#4#6"())

julia> extern(t)
3

julia> t()
ThunkedTangent(var"#5#7"())

julia> t()()
3
```

### When to `@thunk`?
When writing `rrule`s (and to a lesser exent `frule`s), it is important to `@thunk`
appropriately.
Propagation rules that return multiple derivatives may not have all deriviatives used.
 By `@thunk`ing the work required for each derivative, they then compute only what is needed.

#### How do thunks prevent work?
If we have `res = pullback(...) = @thunk(f(x)), @thunk(g(x))`
then if we did `dx + res[1]` then only `f(x)` would be evaluated, not `g(x)`.
Also if we did `ZeroTangent() * res[1]` then the result would be `ZeroTangent()` and `f(x)` would not be evaluated.

#### So why not thunk everything?
`@thunk` creates a closure over the expression, which (effectively) creates a `struct`
with a field for each variable used in the expression, and call overloaded.

Do not use `@thunk` if this would be equal or more work than actually evaluating the expression itself.
This is commonly the case for scalar operators.

For more details see the manual section [on using thunks effectively](http://www.juliadiff.org/ChainRulesCore.jl/dev/writing_good_rules.html#Use-ThunkedTangents-appropriately-1)
"""
struct ThunkedTangent{F} <: AbstractThunk
    f::F
end

(x::ThunkedTangent)() = x.f()
@inline unthunk(x::ThunkedTangent) = x()

Base.show(io::IO, x::ThunkedTangent) = print(io, "ThunkedTangent($(repr(x.f)))")

"""
    InplaceableThunk(val::ThunkedTangent, add!::Function)

A wrapper for a `ThunkedTangent`, that allows it to define an inplace `add!` function.

`add!` should be defined such that: `ithunk.add!(Δ) = Δ .+= ithunk.val`
but it should do this more efficently than simply doing this directly.
(Otherwise one can just use a normal `ThunkedTangent`).

Most operations on an `InplaceableThunk` treat it just like a normal `ThunkedTangent`;
and destroy its inplacability.
"""
struct InplaceableThunk{T<:ThunkedTangent, F} <: AbstractThunk
    val::T
    add!::F
end

unthunk(x::InplaceableThunk) = unthunk(x.val)
(x::InplaceableThunk)() = unthunk(x)

function Base.show(io::IO, x::InplaceableThunk)
    print(io, "InplaceableThunk($(repr(x.val)), $(repr(x.add!)))")
end
