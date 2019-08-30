"""
Subtypes of `AbstractRule` are types which represent the primitive derivative
propagation "rules" that can be composed to implement forward- and reverse-mode
automatic differentiation.

More specifically, a `rule::AbstractRule` is a callable Julia object generally
obtained via calling [`frule`](@ref) or [`rrule`](@ref). Such rules accept
differential values as input, evaluate the chain rule using internally stored/
computed partial derivatives to produce a single differential value, then
return that calculated differential value.

For example:

```jldoctest
julia> using ChainRulesCore: frule, rrule, AbstractRule

julia> x, y = rand(2);

julia> h, dh = frule(hypot, x, y);

julia> h == hypot(x, y)
true

julia> isa(dh, AbstractRule)
true

julia> Δx, Δy = rand(2);

julia> dh(Δx, Δy) == ((x / h) * Δx + (y / h) * Δy)
true

julia> h, (dx, dy) = rrule(hypot, x, y);

julia> h == hypot(x, y)
true

julia> isa(dx, AbstractRule) && isa(dy, AbstractRule)
true

julia> Δh = rand();

julia> dx(Δh) == (x / h) * Δh
true

julia> dy(Δh) == (y / h) * Δh
true
```

See also: [`frule`](@ref), [`rrule`](@ref), [`Rule`](@ref), [`DNERule`](@ref), [`WirtingerRule`](@ref)
"""
abstract type AbstractRule end

# this ensures that consumers don't have to special-case rule destructuring
Base.iterate(rule::AbstractRule) = (rule, nothing)
Base.iterate(::AbstractRule, ::Any) = nothing

# This ensures we don't need to check whether the result of `rrule`/`frule` is a tuple
# in order to get the `i`th rule (assuming it's 1)
Base.getindex(rule::AbstractRule, i::Integer) = i == 1 ? rule : throw(BoundsError())

"""
    accumulate(Δ, rule::AbstractRule, args...)

Return `Δ + rule(args...)` evaluated in a manner that supports ChainRulesCore'
various `AbstractDifferential` types.

This method intended to be customizable for specific rules/input types. For
example, here is pseudocode to overload `accumulate` w.r.t. a specific forward
differentiation rule for a given function `f`:

```
df(x) = # forward differentiation primitive implementation

frule(::typeof(f), x) = (f(x), Rule(df))

accumulate(Δ, rule::Rule{typeof(df)}, x) = # customized `accumulate` implementation
```

See also: [`accumulate!`](@ref), [`store!`](@ref), [`AbstractRule`](@ref)
"""
accumulate(Δ, rule::AbstractRule, args...) = Δ + rule(args...)

"""
    accumulate!(Δ, rule::AbstractRule, args...)

Similar to [`accumulate`](@ref), but compute `Δ + rule(args...)` in-place,
storing the result in `Δ`.

Note that this function internally calls `Base.Broadcast.materialize!(Δ, ...)`.

See also: [`accumulate`](@ref), [`store!`](@ref), [`AbstractRule`](@ref)
"""
function accumulate!(Δ, rule::AbstractRule, args...)
    return materialize!(Δ, broadcastable(cast(Δ) + rule(args...)))
end

accumulate!(Δ::Number, rule::AbstractRule, args...) = accumulate(Δ, rule, args...)

"""
    store!(Δ, rule::AbstractRule, args...)

Compute `rule(args...)` and store the result in `Δ`, potentially avoiding
intermediate temporary allocations that might be necessary for alternative
approaches (e.g. `copyto!(Δ, extern(rule(args...)))`)

Note that this function internally calls `Base.Broadcast.materialize!(Δ, ...)`.

Like [`accumulate`](@ref) and [`accumulate!`](@ref), this function is intended
to be customizable for specific rules/input types.

See also: [`accumulate`](@ref), [`accumulate!`](@ref), [`AbstractRule`](@ref)
"""
store!(Δ, rule::AbstractRule, args...) = materialize!(Δ, broadcastable(rule(args...)))

#####
##### `Rule`
#####


"""
    Rule(propation_function[, updating_function])

Return a `Rule` that wraps the given `propation_function`. It is assumed that
`propation_function` is a callable object whose arguments are differential
values, and whose output is a single differential value calculated by applying
internally stored/computed partial derivatives to the input differential
values.

If an updating function is provided, it is assumed to have the signature `u(Δ, xs...)`
and to store the result of the propagation function applied to the arguments `xs` into
`Δ` in-place, returning `Δ`.

For example:

```
frule(::typeof(*), x, y) = x * y, Rule((Δx, Δy) -> Δx * y + x * Δy)

rrule(::typeof(*), x, y) = x * y, (Rule(ΔΩ -> ΔΩ * y'), Rule(ΔΩ -> x' * ΔΩ))
```

See also: [`frule`](@ref), [`rrule`](@ref), [`accumulate`](@ref), [`accumulate!`](@ref), [`store!`](@ref)
"""
struct Rule{F,U<:Union{Function,Nothing}} <: AbstractRule
    f::F
    u::U
end

# NOTE: Using `Core.Typeof` instead of `typeof` here so that if we define a rule for some
# constructor based on a `UnionAll`, we get `Rule{Type{Thing}}` instead of `Rule{UnionAll}`
Rule(f) = Rule{Core.Typeof(f),Nothing}(f, nothing)

(rule::Rule)(args...) = rule.f(args...)

Base.show(io::IO, rule::Rule{<:Any, Nothing}) = print(io, "Rule($(rule.f))")
Base.show(io::IO, rule::Rule) = print(io, "Rule($(rule.f), $(rule.u))")

# Specialized accumulation
# TODO: Does this need to be overdubbed in the rule context?
accumulate!(Δ, rule::Rule{F,U}, args...) where {F,U<:Function} = rule.u(Δ, args...)

#####
##### `DNERule`
#####

"""
    DNERule(args...)

Construct a `DNERule` object, which is an `AbstractRule` that signifies that the
current function is not differentiable with respect to a particular parameter.
**DNE** is an abbreviation for Does Not Exist.
"""
struct DNERule <: AbstractRule end

DNERule(args...) = DNE()

#####
##### `WirtingerRule`
#####

"""
    WirtingerRule(primal::AbstractRule, conjugate::AbstractRule)

Construct a `WirtingerRule` object, which is an `AbstractRule` that consists of
an `AbstractRule` for both the primal derivative ``∂/∂x`` and the conjugate
derivative ``∂/∂x̅``. If the domain `𝒟` of the function might be real, consider
calling `AbstractRule(𝒟, primal, conjugate)` instead, to make use of a more
efficient representation wherever possible.
"""
struct WirtingerRule{P<:AbstractRule,C<:AbstractRule} <: AbstractRule
    primal::P
    conjugate::C
end

function (rule::WirtingerRule)(args...)
    return Wirtinger(rule.primal(args...), rule.conjugate(args...))
end

"""
    AbstractRule(𝒟::Type, primal::AbstractRule, conjugate::AbstractRule)

Return a `Rule` evaluating to `primal(Δ) + conjugate(Δ)` if `𝒟 <: Real`,
otherwise return `WirtingerRule(P, C)`.
"""
function AbstractRule(𝒟::Type, primal::AbstractRule, conjugate::AbstractRule)
    if 𝒟 <: Real || eltype(𝒟) <: Real
        return Rule((args...) -> (primal(args...) + conjugate(args...)))
    else
        return WirtingerRule(primal, conjugate)
    end
end
