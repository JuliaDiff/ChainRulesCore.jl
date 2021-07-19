# Opting out of rules

It is common to define rules fairly generically.
Often matching (or exceeding) how generic the matching original primal method is.
Sometimes this is not the correct behavour.
Sometimes the AD can do better than this human defined rule.
If this is generally the case, then we should not have the rule defined at all.
But if it is only the case for a particular set of types, then we want to opt-out just that one.
This is done with the [`@opt_out`](@ref) macro.

Consider one might have a rrule for `sum` (the following simplified from the one in [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/master/src/rulesets/Base/mapreduce.jl) itself)
```julia
function rrule(::typeof(sum), x::AbstractArray{<:Number}; dims=:)
    y = sum(x; dims=dims)
    project = ProjectTo(x)
    function sum_pullback(ȳ)
        # broadcasting the two works out the size no-matter `dims`
        # project makes sure we stay in the same vector subspace as `x`
        # no putting in off-diagonal entries in Diagonal etc
        x̄ = project(broadcast(last∘tuple, x, ȳ)))
        return (NoTangent(), x̄)
    end
    return y, sum_pullback
end
```

That is a fairly reasonable `rrule` for the vast majority of cases.

You might have a custom array type for which you could write a faster rule.
For example, the pullback for summing a`SkewSymmetric` matrix can be optimizes to basically be `Diagonal(fill(ȳ, size(x,1)))`.
To do that, you can indeed write another more specific [`rrule`](@ref).
But another case is where the AD system itself would generate a more optimized case.

For example, the a [`NamedDimArray`](https://github.com/invenia/NamedDims.jl) is a thin wrapper around some other array type.
It's sum method is basically just to call `sum` on it's parent.
It is entirely conceivable[^1] that the AD system can do better than our `rrule` here.
For example by avoiding the overhead of [`project`ing](@ref ProjectTo).

To opt-out of using the `rrule` and to allow the AD system to do its own thing we use the
[`@opt_out`](@ref) macro, to say to not use it for sum.

```julia
@opt_out rrule(::typeof(sum), ::NamedDimsArray)
```

We could even opt-out for all 1 arg functions.
```@julia
@opt_out rrule(::Any, ::NamedDimsArray)
```
Though this is likely to cause some method-ambiguities.

Similar can be done  `@opt_out frule`.
It can also be done passing in a [`RuleConfig`](@ref config).


### How to support this (for AD implementers)

We provide two ways to know that a rule has been opted out of.

## `rrule` / `frule` returns `nothing`

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
The Julia compiler, will specialize based on inferring the restun type of `rrule`, and so can remove that branch.

## `no_rrule` / `no_frule` has a method

`@opt_out` also defines a method for  [`ChainRulesCore.no_frule`](@ref) or [`ChainRulesCore.no_rrule`](@ref).
The use of this method doesn't matter, what matters is it's method-table.
A simple thing you can do with this is not support opting out.
To do this, filter all methods from the `rrule`/`frule` method table that also occur in the `no_frule`/`no_rrule` table.
This will thus avoid ever hitting an `rrule`/`frule` that returns `nothing` and thus makes your library error.
This is easily done, though it does mean ignoring the user's stated desire to opt out of the rule.

More complex you can use this to generate code that triggers your AD.
If for a given signature there is a more specific method in the `no_rrule`/`no_frule` method-table, than the one that would be hit from the `rrule`/`frule` table
(Excluding the one that exactly matches which will return `nothing`) then you know that the rule should not be used.
You can, likely by looking at the primal method table, workout which method you would have it if the rule had not been defined,
and then `invoke` it.



[^1]: It is also possible, that this is not the case. Benchmark your real uses cases.