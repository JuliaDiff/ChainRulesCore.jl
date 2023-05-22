# [Opting out of rules](@id opt_out)

It is common to define rules fairly generically.
Often matching (or exceeding) how generic the matching original primal method is.
Sometimes this is not the correct behaviour.
Sometimes the AD can do better than this human defined rule.
If this is generally the case, then we should not have the rule defined at all.
But if it is only the case for a particular set of types, then we want to opt-out just that one.
This is done with the [`@opt_out`](@ref) macro.

Consider one a `rrule` for `sum` (the following simplified from the one in [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/mapreduce.jl) itself)
```julia
function rrule(::typeof(sum), x::AbstractArray{<:Number})
    y = sum(x; dims=dims)
    project = ProjectTo(x)
    function sum_pullback(ȳ)
        x̄ = project(fill(ȳ, size(x)))
        return (NoTangent(), x̄)
    end
    return y, sum_pullback
end
```

That is a fairly reasonable `rrule` for the vast majority of cases.
You might have a custom array type for which you could write a faster rule.
In which case you would do that, by writing a faster, more specific, `rrule`.
But sometimes, it is the case that ADing the (faster, more specific) primal for your custom array type would yeild the faster pullback without you having to write a `rrule` by hand.

Consider a summing  [`SkewSymmetric` (anti-symmetric)](https://en.wikipedia.org/wiki/Skew-symmetric_matrix) matrix.
The skew symmetric matrix has structural zeros on the diagonal, and off-diagonals are paired with their negation.
Thus the sum is always going to be zero.
As such the author of that matrix type would probably have overloaded `sum(x::SkewSymmetric{T}) where T = zero(T)`.
ADing this would result in the tangent computed for `x` as `ZeroTangent()` and it would be very fast since AD can see that `x` is never used in the right-hand side.
In contrast the generic method for `AbstractArray` defined above would have to allocate the fill array, and then compute the skew projection.
Only to find out the output would be projected to `SkewSymmetric(zeros(T))` anyway (slower, and a less useful type).

To opt-out of using the generic `rrule` and to allow the AD system to do its own thing we use the
[`@opt_out`](@ref) macro, to say to not use it for sum of `SkewSymmetric`.

```julia
@opt_out rrule(::typeof(sum), ::SkewSymmetric)
```

Perhaps we might not want to ever use rules for SkewSymmetric, because we have determined that it is always better to leave it to the AD, unless a very specific rule has been written[^1].
We could then opt-out for all 1 arg functions.
```@julia
@opt_out rrule(::Any, ::SkewSymmetric)
```
Though this is likely to cause some method-ambiguities, if we do it for more, but we can resolve those.


Similar can be done  `@opt_out frule`.
It can also be done passing in a [`RuleConfig`](@ref config).


!!! warning "If the general rule uses a config, the opt-out must also"
    Following the same principles as for [rules with config](@ref config), a rule with a `RuleConfig` argument will take precedence over one without, including if that one is a opt-out rule.
    But if the general rule does not use a config, then the opt-out rule *can* use a config.
    This allows, for example, you to use opt-out to avoid a particular AD system using a opt-out rule that takes that particular AD's config.
    
[^1]: seems unlikely, but it is possible, there is a lot of structure that can be taken advantage of for some matrix types.
