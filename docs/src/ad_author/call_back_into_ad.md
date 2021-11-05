# Declaring support for calling back into ADs

To declare support or lack of support for forward and reverse-mode, use the two pairs of complementary types.
For reverse mode: [`HasReverseMode`](@ref), [`NoReverseMode`](@ref).
For forwards mode: [`HasForwardsMode`](@ref), [`NoForwardsMode`](@ref).
AD systems that support any calling back into AD should have one from each set.

If an AD `HasReverseMode`, then it must define [`rrule_via_ad`](@ref) for that RuleConfig subtype.
Similarly, if an AD `HasForwardsMode` then it must define [`frule_via_ad`](@ref) for that RuleConfig subtype.

For example:
```julia
struct MyReverseOnlyADRuleConfig <: RuleConfig{Union{HasReverseMode, NoForwardsMode}} end

function ChainRulesCore.rrule_via_ad(::MyReverseOnlyADRuleConfig, f, args...)
    ...
    return y, pullback
end
```

Note that it is not actually required that the same AD is used for forward and reverse.
For example [Nabla.jl](https://github.com/invenia/Nabla.jl/) is a reverse mode AD.
It might declare that it `HasForwardsMode`, and then define a wrapper around [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) in order to provide that capacity.
