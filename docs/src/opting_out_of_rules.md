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
Only to findout the output would be projected to `SkewSymmetric(zeros(T))` anyway (slower, and a less useful type).

To opt-out of using the generic `rrule` and to allow the AD system to do its own thing we use the
[`@opt_out`](@ref) macro, to say to not use it for sum of `SkewSymmetric`.

```julia
@opt_out rrule(::typeof(sum), ::SkewSymmetric)
```

Perhaps we might not want to ever use rules for SkewSymmetric, because we have determined that it is always better to leave it to the AD, unless a verys specific rule has been written[^1].
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
    

## How to support this (for AD implementers)

We provide two ways to know that a rule has been opted out of.

### `rrule` / `frule` returns `nothing`

`@opt_out` defines a `frule` or `rrule` matching the signature that returns `nothing`.

If you are in a position to generate code, in response to values returned by function calls then you can do something like:
```@julia
res = rrule(f, xs)
if res === nothing
    y, pullback = perform_ad_via_decomposition(r, xs)  # do AD without hitting the rrule
else
    y, pullback = res
end
```
The Julia compiler will specialize based on inferring the return type of `rrule`, and so can remove that branch.

### `no_rrule` / `no_frule` has a method

`@opt_out` also defines a method for  [`ChainRulesCore.no_frule`](@ref) or [`ChainRulesCore.no_rrule`](@ref).
The body of this method doesn't matter, what matters is that it is a method-table.
A simple thing you can do with this is not support opting out.
To do this, filter all methods from the `rrule`/`frule` method table that also occur in the `no_frule`/`no_rrule` table.
This will thus avoid ever hitting an `rrule`/`frule` that returns `nothing` (and thus prevents your library from erroring).
This is easily done, though it does mean ignoring the user's stated desire to opt out of the rule.

More complex you can use this to generate code that triggers your AD.
If for a given signature there is a more specific method in the `no_rrule`/`no_frule` method-table, than the one that would be hit from the `rrule`/`frule` table
(Excluding the one that exactly matches which will return `nothing`) then you know that the rule should not be used.
You can, likely by looking at the primal method table, workout which method you would have it if the rule had not been defined,
and then `invoke` it.

[^1]: seems unlikely, but it is possible, there is a lot of structure that can be taken advantage of for some matrix types.
