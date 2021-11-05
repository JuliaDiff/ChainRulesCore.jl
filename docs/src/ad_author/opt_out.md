# Support opting out of rules

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
