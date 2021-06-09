# [Rule configurations and calling back into AD](@id config)

[`RuleConfig`](@ref) is a method for making rules conditionally defined based on the presence of certain features in the AD system.
Once key such feature is the ability to perform AD either in forwards or reverse mode or both.

This is done with a trait-like system (not Holy Traits), where the `RuleConfig` has a union of types as its only type-parameter.
Where each type represents a particular special feature of this AD.
To indicate that the AD system has a special property, its `RuleConfig` should be defined as:
```julia
struct MyADRuleConfig <: RuleConfig{Union{Feature1, Feature2}} end
```
And rules that should only be defined when an AD has a particular special property write:
```julia
rrule(::RuleConfig{>:Feature1}, f, args...) = # rrule that should only be define for ADs with `Feature1`

frule(::RuleConfig{>:Union{Feature1,Feature2}}, f, args...) = # frule that should only be define for ADs with both `Feature1` and `Feature2`
```

A prominent use of this is in declaring that the AD system can, or cannot support being called from within the rule definitions.

## Declaring support for calling back into ADs

To declare support or lack of support for forward and reverse-mode, use the two pairs of complementary types.
For reverse mode: [`CanReverseMode`](@ref), [`NoReverseMode`](@ref).
For forwards mode: [`CanForwardsMode`](@ref), [`NoForwardsMode`](@ref).
AD systems that support any calling back into AD should have one from each set.

If an AD `CanReverseMode`, then it must define [`rrule_via_ad`](@ref) for that RuleConfig subtype.
Similarly, if an AD `CanForwardsMode` then it must define [`frule_via_ad`](@ref) for that RuleConfig subtype.

For example:
```julia
struct MyReverseOnlyADRuleConfig <: RuleConfig{Union{CanReverseMode, NoForwardsMode}} end

function ChainRulesCore.rrule_via_ad(::MyReverseOnlyADRuleConfig, f, args...)
    ...
    return y, pullback
end
```

Note that it is not actually required that the same AD is used for forward and reverse.
For example [Nabla.jl](https://github.com/invenia/Nabla.jl/) is a reverse mode AD.
It might declare that it `CanForwardsMode`, and then define a wrapper around [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) in order to provide that capacity.

## Writing rules that call back into AD

To define e.g. rules for higher order functions, it is useful to be able to call back into the AD system to get it to do some work for you.

For example the rule for reverse mode AD for `map` might like to use forward mode AD if one is available.
Particularly for the case where only a single input collection is being mapped over.
In that case we know the most efficient way to compute that sub-program is in forwards, as each call with-in the map only takes a single input.

Note: the following is not the most efficient rule for `map` via forward, but attempts to be clearer for demonstration purposes.

```julia
function rrule(config::RuleConfig{>:CanFowardsMode}, ::typeof(map), f::Function, x::Array{<:Real})
    y_and_ḟ_and_ẏ = map(x) do xi
        frule_via_ad(config, one(xi), f, xi)
    end
    ḟ_and_ẏ = last.(y_and_ḟ_and_ẏ)

    function pullback_map(ȳ)
        ḟ = first.(ḟ_and_ẏ)
        ẏ = last.(ḟ_and_ẏ)
        return NoTangent(), ȳ * sum(ḟ), ȳ .* ẏ
    end

    return first.(y_and_ḟ_and_ẏ), pullback_map
end
```

TODO demo for reverse.

## Writing rules that depend on other special requirements of the AD.

As of right now there are no such special properties defined.
It is likely that in the future such will be provided for e.g. mutation support.

Such a thing would look like:
```julia
struct SupportsMutation end

function rrule(
    ::RuleConfig{>:SupportsMutatation}, typeof(push!), x::Vector
)
    y = push!(x)

    function push!_pullback(ȳ)
        pop!(x)  # undo change to primal incase it is used in another pullback we haven't called yet
        pop!(ȳ)  # accumulate gradient via mutating ȳ, then return ZeroTangent
        return NoTangent(), ZeroTangent()
    end

    return y, push!_pullback
end
```
and it would be used in the AD e.g. as follows:
```julia
struct EnzymeRuleConfig <: RuleConfig{Union{SupportsMutation, CanReverseMode, NoForwardsMode}}
```

Note: you can only depend on the presence of a feature, not its absence.
This means we may need to define features and their compliments, when one is not the obvious default (as in the fast of [`CanReverseMode`](@ref)/[`NoReverseMode`](@ref) and [`CanForwardsMode`](@ref)/[`NoForwardsMode`](@ref).).

## Writing rules that are only for your own AD

A special case of the above is writing rules that are defined only for your own AD.
Rules which otherwise would be type-piracy, and would affect other AD systems.
This could be done via making up a special property type and dispatching on it.
But there is no need, as we can dispatch on the `RuleConfig` subtype directly.

For example in order to avoid mutation in nested AD situations, Zygote might want to have a rule for [`add!!`](@ref) that makes it just do `+`.

```julia
struct ZygoteConfig <: RuleConfig{Union{}} end

rrule(::ZygoteConfig, typeof(ChainRulesCore.add!!), a, b) = a+b, Δ->(NoTangent(), Δ, Δ)
```
