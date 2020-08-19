# Using ChainRules in your AutoDiff system

This section is for authors of AD systems.
It assumes a pretty solid understanding of Julia, and of Automatic Differentiation.
It explains how to make use of ChainRule's rule-sets,
to avoid having to code all your own AD primitives / custom sensitives.

There are 3 main ways to use ChainRules in your AutoDiff system.

1. [Operation Overloading Generation](autodiff/operator_overloading)
  - This is primarily intended for operator overloading based AD systems which will generate overloads for primal function based for their overloaded types based on the existance of an `rrule`/`frule`.
  - A source code generation based AD can also use this by overloading their tranform generating function directly so as not to generate a tranform but to just return a result.
  - This does not play nice with Revise.jl, modifying or especially deleting rules may not be reflected.
2. Source code transform based on `rrule`/`frule` method-table
  - Always use `rrule`/`frule` iff and only if use the rules that exist, else generate normal AD path.
  - This avoids having branches in your generated code.
  - This requires maintaining your own back-edges
  - This is pretty hard-code even by the standard of source code tranformations
3. Source code tranform based on inserting branches that check of `rrule`/`frule` return `nothing`
  - if the `rrule`/`frule` returns a rule result then use it, if it return `nothing` then do normal AD path 
  - In theory type inference optimizes these branchs out; in practice it may not.
  - This is a fairly simple Cassette overdub of all calls, and is suitable for overloading based AD or source code transformation.
