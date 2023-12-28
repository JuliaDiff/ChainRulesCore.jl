# [Rule configurations and calling back into AD](@id config)

[`RuleConfig`](@ref) is a method for making rules conditionally defined based on the presence of certain features in the AD system.
One key such feature is the ability to perform AD either in forwards or reverse mode or both.

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

!!! warning "Rules with Config always take precedence over rules without"
    Even if the other arguments are more specific the rule with the config will always take precedence.
    For example of there is a rule `rrule(::RuleConfig, ::typeof(foo), ::Any)` and other `rrule(foo, ::Float64)`,
    the first will always be selected.
    This is because the AD will always attempt to provide its config when checking for a rule, and only if that doesn't match, will the config-less rule be tried.
    In practice this doesn't happen often, but when it does the solution is a little ugly -- though very similar to resolving method ambiguities.  
    You need to manually add methods that dispatch from a rule with config to the one without.
    See for example the [rule for `sum(abs2, xs)` in ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/4ad975826ea0639ad709aeb36cc5051b6bf82eaa/src/rulesets/Base/mapreduce.jl#L87-L113).



A prominent use of this is in declaring that the AD system can, or cannot support being called from within the rule definitions.

## Writing rules that call back into AD

To define e.g. rules for higher order functions, it is useful to be able to call back into the AD system to get it to do some work for you.

For example the rule for reverse mode AD for `map` might like to use forward mode AD if one is available.
Particularly for the case where only a single input collection is being mapped over.
In that case we know the most efficient way to compute that sub-program is in forwards, as each call with-in the map only takes a single input.

Note: the following is not the most efficient rule for `map` via forward, but attempts to be clearer for demonstration purposes.

```julia
function rrule(config::RuleConfig{>:HasForwardsMode}, ::typeof(map), f::Function, x::Array{<:Real})
    # real code would support functors/closures, but in interest of keeping example short we exclude it:
    @assert (fieldcount(typeof(f)) == 0) "Functors/Closures are not supported"

    y_and_ẏ = map(x) do xi
        frule_via_ad(config, (NoTangent(), one(xi)), f, xi)
    end
    y = first.(y_and_ẏ)
    ẏ = last.(y_and_ẏ)

    pullback_map(ȳ) = NoTangent(), NoTangent(), ȳ .* ẏ
    return y, pullback_map
end
```

## Writing rules that depend on other special requirements of the AD.

The `>:HasReverseMode` and `>:HasForwardsMode` are two examples of special properties that a `RuleConfig` could allow.
Others could also exist, but right now they are the only two.
It is likely that in the future other such will be provided 

Note: you can only depend on the presence of a feature, not its absence.
This means we may need to define features and their complements, when one is not the obvious default (as in the case of [`HasReverseMode`](@ref)/[`NoReverseMode`](@ref) and [`HasForwardsMode`](@ref)/[`NoForwardsMode`](@ref).).

Such special properties generally should only be defined in `ChainRulesCore`.
(Theoretically, they could be defined elsewhere, but the AD and the package containing the rule need to load them, and ChainRulesCore is the place for things like that.)


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

As an alternative to rules only for one AD, would be to add new special property definitions to ChainRulesCore (as described above) which would capture what makes that AD special.
