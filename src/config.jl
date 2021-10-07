"""
    RuleConfig{T}

The configuration for what rules to use.
`T`: **traits**. This should be a `Union` of all special traits needed for rules to be
allowed to be defined for your AD. If nothing special this should be set to `Union{}`.

**AD authors** should define a subtype of `RuleConfig` to use when calling `frule`/`rrule`.

**Rule authors** can dispatch on this config when defining rules.
For example:
```julia
# only define rrule for `pop!` on AD systems where mutation is supported.
rrule(::RuleConfig{>:SupportsMutation}, typeof(pop!), ::Vector) = ...

# this definition of map is for any AD that defines a forwards mode
rrule(conf::RuleConfig{>:HasForwardsMode}, typeof(map), ::Vector) = ...

# this definition of map is for any AD that only defines a reverse mode.
# It is not as good as the rrule that can be used if the AD defines a forward-mode as well.
rrule(conf::RuleConfig{>:Union{NoForwardsMode, HasReverseMode}}, typeof(map), ::Vector) = ...
```

For more details see [rule configurations and calling back into AD](@ref config).
"""
abstract type RuleConfig{T} end

# Broadcast like a scalar
Base.Broadcast.broadcastable(config::RuleConfig) = Ref(config)

abstract type ReverseModeCapability end

"""
HasReverseMode

This trait indicates that a `RuleConfig{>:HasReverseMode}` can perform reverse mode AD.
If it is set then [`rrule_via_ad`](@ref) must be implemented.
"""
struct HasReverseMode <: ReverseModeCapability end

"""
NoReverseMode

This is the complement to [`HasReverseMode`](@ref). To avoid ambiguities [`RuleConfig`]s
that do not support performing reverse mode AD should be `RuleConfig{>:NoReverseMode}`.
"""
struct NoReverseMode <: ReverseModeCapability end

abstract type ForwardsModeCapability end

"""
HasForwardsMode

This trait indicates that a `RuleConfig{>:HasForwardsMode}` can perform forward mode AD.
If it is set then [`frule_via_ad`](@ref) must be implemented.
"""
struct HasForwardsMode <: ForwardsModeCapability end

"""
NoForwardsMode

This is the complement to [`HasForwardsMode`](@ref). To avoid ambiguities [`RuleConfig`]s
that do not support performing forwards mode AD should be `RuleConfig{>:NoForwardsMode}`.
"""
struct NoForwardsMode <: ForwardsModeCapability end

"""
    frule_via_ad(::RuleConfig{>:HasForwardsMode}, ȧrgs, f, args...; kwargs...)

This function has the same API as [`frule`](@ref), but operates via performing forwards mode
automatic differentiation.
Any `RuleConfig` subtype that supports the [`HasForwardsMode`](@ref) special feature must
provide an implementation of it.

See also: [`rrule_via_ad`](@ref), [`RuleConfig`](@ref) and the documentation on
[rule configurations and calling back into AD](@ref config)
"""
function frule_via_ad end

"""
    rrule_via_ad(::RuleConfig{>:HasReverseMode}, f, args...; kwargs...)

This function has the same API as [`rrule`](@ref), but operates via performing reverse mode
automatic differentiation.
Any `RuleConfig` subtype that supports the [`HasReverseMode`](@ref) special feature must
provide an implementation of it.

See also: [`frule_via_ad`](@ref), [`RuleConfig`](@ref) and the documentation on
[rule configurations and calling back into AD](@ref config)
"""
function rrule_via_ad end
