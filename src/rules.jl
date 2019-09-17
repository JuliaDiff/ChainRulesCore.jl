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
        fx, df = (f(x), (_, Δx) -> ForwardDiff.derivative(f, x) * Δx)
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

    (Ω, (ṡelf, ẋ₁, ẋ₂, ...) -> Ω̇₁, Ω̇₂, ...)

The second return value is the propagation rule, or the pushforward.
It takes in differentials corresponding to the inputs (`ẋ₁, ẋ₂, ...`)
and `ṡelf` the internal values of the function (for closures).


If no method matching `frule(f, xs...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```
julia> x = rand();

julia> sinx, sin_pushforward = frule(sin, x);

julia> sinx == sin(x)
true

julia> sin_pushforward(NamedTuple(), 1) == cos(x)
true
```

unary input, binary output scalar function:

```
julia> x = rand();

julia> sincosx, sincos_pushforward = frule(sincos, x);

julia> sincosx == sincos(x)
true

julia> sincos_pushforward(NamedTuple(), 1) == (cos(x), -sin(x))
true
```

See also: [`rrule`](@ref), [`@scalar_rule`](@ref)
"""
frule(::Any, ::Vararg{Any}; kwargs...) = nothing

"""
    rrule(f, x...)

Expressing `x` as the tuple `(x₁, x₂, ...)` and the output tuple of `f(x...)`
as `Ω`, return the tuple:

    (Ω, (Ω̄₁, Ω̄₂, ...) -> (s̄elf, x̄₁, x̄₂, ...))

Where the second return value is the the propagation rule or pullback.
It takes in differentials corresponding to the outputs (`x̄₁, x̄₂, ...`),
and `s̄elf`, the internal values of the function itself (for closures)

If no method matching `rrule(f, xs...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```
julia> x = rand();

julia> sinx, sin_pullback = rrule(sin, x);

julia> sinx == sin(x)
true

julia> sin_pullback(1) == (NO_FIELDS, cos(x))
true
```

binary input, unary output scalar function:

```
julia> x, y = rand(2);

julia> hypotxy, hypot_pullback = rrule(hypot, x, y);

julia> hypotxy == hypot(x, y)
true

julia> hypot_pullback(1) == (NO_FIELDS, (x / hypot(x, y)), (y / hypot(x, y)))
true
```

See also: [`frule`](@ref), [`@scalar_rule`](@ref)
"""
rrule(::Any, ::Vararg{Any}; kwargs...) = nothing
