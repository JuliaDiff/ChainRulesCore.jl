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
