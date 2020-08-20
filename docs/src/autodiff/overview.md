# Using ChainRules in your AutoDiff system

This section is for authors of AD systems.
It assumes a pretty solid understanding of Julia, and of automatic differentiation.
It explains how to make use of ChainRule's rule sets,
to avoid having to code all your own AD primitives / custom sensitives.

There are 3 main ways to access ChainRules rule sets in your AutoDiff system.

1. [Operation Overloading Generation](autodiff/operator_overloading)
  - This is primarily intended for operator overloading based AD systems which will generate overloads for primal function based for their overloaded types based on the existance of an `rrule`/`frule`.
  - A source code generation based AD can also use this by overloading their transform generating function directly so as not to recursively generate a transform but to just return the rule.
  - This does not play nice with Revise.jl, adding or modifying rules in loaded files will not be reflected until a manual refresh, and deleting rules will not be reflected at all.
2. Source code tranform based on inserting branches that check of `rrule`/`frule` return `nothing`
  - if the `rrule`/`frule` returns a rule result then use it, if it return `nothing` then do normal AD path 
  - In theory type inference optimizes these branchs out; in practice it may not.
  - This is a fairly simple Cassette overdub (or similar) of all calls, and is suitable for overloading based AD or source code transformation.
3. Source code transform based on `rrule`/`frule` method-table
  - Always use `rrule`/`frule` iff and only if use the rules that exist, else generate normal AD path.
  - This avoids having branches in your generated code.
  - This requires maintaining your own back-edges
  - This is pretty hard-code even by the standard of source code tranformations
