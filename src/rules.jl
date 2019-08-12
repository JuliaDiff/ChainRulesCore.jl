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
accumulate(Δ, rule::AbstractRule, args...) = add(Δ, rule(args...))

"""
    accumulate!(Δ, rule::AbstractRule, args...)

Similar to [`accumulate`](@ref), but compute `Δ + rule(args...)` in-place,
storing the result in `Δ`.

Note that this function internally calls `Base.Broadcast.materialize!(Δ, ...)`.

See also: [`accumulate`](@ref), [`store!`](@ref), [`AbstractRule`](@ref)
"""
function accumulate!(Δ, rule::AbstractRule, args...)
    return materialize!(Δ, broadcastable(add(cast(Δ), rule(args...))))
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

Cassette.@context RuleContext

const RULE_CONTEXT = Cassette.disablehooks(RuleContext())

Cassette.overdub(::RuleContext, ::typeof(+), a, b) = add(a, b)
Cassette.overdub(::RuleContext, ::typeof(*), a, b) = mul(a, b)

Cassette.overdub(::RuleContext, ::typeof(add), a, b) = add(a, b)
Cassette.overdub(::RuleContext, ::typeof(mul), a, b) = mul(a, b)

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

(rule::Rule{F})(args...) where {F} = Cassette.overdub(RULE_CONTEXT, rule.f, args...)

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
TODO
"""
struct WirtingerRule{P<:AbstractRule,C<:AbstractRule} <: AbstractRule
    primal::P
    conjugate::C
end

function (rule::WirtingerRule)(args...)
    return Wirtinger(rule.primal(args...), rule.conjugate(args...))
end

#####
##### `frule`/`rrule`
#####

#=
In some weird ideal sense, the fallback for e.g. `frule` should actually be "get
the derivative via forward-mode AD". This is necessary to enable mixed-mode
rules, where e.g. `frule` is used within a `rrule` definition. For example,
broadcasted functions may not themselves be forward-mode *primitives*, but are
often forward-mode *differentiable*.

ChainRulesCore, by design, is decoupled from any specific AD implementation. How,
then, do we know which AD to fall back to when there isn't a primitive defined?

Well, if you're a greedy AD implementation, you can just overload `frule` and/or
`rrule` to use your AD directly. However, this won't play nice with other AD
packages doing the same thing, and thus could cause load-order-dependent
problems for downstream users.

It turns out, Cassette solves this problem nicely by allowing AD authors to
overload the fallbacks w.r.t. their own context. Example using ForwardDiff:

```
using ChainRulesCore, ForwardDiff, Cassette

Cassette.@context MyChainRuleCtx

# ForwardDiff, itself, can call `my_frule` instead of
# `frule` to utilize the ForwardDiff-injected ChainRulesCore
# infrastructure
my_frule(args...) = Cassette.overdub(MyChainRuleCtx(), frule, args...)

function Cassette.execute(::MyChainRuleCtx, ::typeof(frule), f, x::Number)
    r = frule(f, x)
    if isa(r, Nothing)
        fx, df = (f(x), Rule(Δx -> ForwardDiff.derivative(f, x) * Δx))
    else
        fx, df = r
    end
    return fx, df
end
```
=#

"""
    frule(f, x...)

Expressing `x` as the tuple `(x₁, x₂, ...)` and the output tuple of `f(x...)`
as `Ω`, return the tuple:

    (Ω, (rule_for_ΔΩ₁::AbstractRule, rule_for_ΔΩ₂::AbstractRule, ...))

where each returned propagation rule `rule_for_ΔΩᵢ` can be invoked as

    rule_for_ΔΩᵢ(Δx₁, Δx₂, ...)

to yield `Ωᵢ`'s corresponding differential `ΔΩᵢ`. To illustrate, if all involved
values are real-valued scalars, this differential can be written as:

    ΔΩᵢ = ∂Ωᵢ_∂x₁ * Δx₁ + ∂Ωᵢ_∂x₂ * Δx₂ + ...

If no method matching `frule(f, xs...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```jldoctest
julia> x = rand();

julia> sinx, dsin = frule(sin, x);

julia> sinx == sin(x)
true

julia> dsin(1) == cos(x)
true
```

unary input, binary output scalar function:

```jldoctest
julia> x = rand();

julia> sincosx, (dsin, dcos) = frule(sincos, x);

julia> sincosx == sincos(x)
true

julia> dsin(1) == cos(x)
true

julia> dcos(1) == -sin(x)
true
```

See also: [`rrule`](@ref), [`AbstractRule`](@ref), [`@scalar_rule`](@ref)
"""
frule(::Any, ::Vararg{Any}; kwargs...) = nothing

"""
    rrule(f, x...)

Expressing `x` as the tuple `(x₁, x₂, ...)` and the output tuple of `f(x...)`
as `Ω`, return the tuple:

    (Ω, (rule_for_Δx₁::AbstractRule, rule_for_Δx₂::AbstractRule, ...))

where each returned propagation rule `rule_for_Δxᵢ` can be invoked as

    rule_for_Δxᵢ(ΔΩ₁, ΔΩ₂, ...)

to yield `xᵢ`'s corresponding differential `Δxᵢ`. To illustrate, if all involved
values are real-valued scalars, this differential can be written as:

    Δxᵢ = ∂Ω₁_∂xᵢ * ΔΩ₁ + ∂Ω₂_∂xᵢ * ΔΩ₂ + ...

If no method matching `rrule(f, xs...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```jldoctest
julia> x = rand();

julia> sinx, dx = rrule(sin, x);

julia> sinx == sin(x)
true

julia> dx(1) == cos(x)
true
```

binary input, unary output scalar function:

```jldoctest
julia> x, y = rand(2);

julia> hypotxy, (dx, dy) = rrule(hypot, x, y);

julia> hypotxy == hypot(x, y)
true

julia> dx(1) == (x / hypot(x, y))
true

julia> dy(1) == (y / hypot(x, y))
true
```

See also: [`frule`](@ref), [`AbstractRule`](@ref), [`@scalar_rule`](@ref)
"""
rrule(::Any, ::Vararg{Any}; kwargs...) = nothing
