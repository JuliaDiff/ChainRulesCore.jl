# Using ChainRules in your AD system

This section is for authors of AD systems.
It assumes a pretty solid understanding of both Julia and automatic differentiation.
It explains how to make use of ChainRule's "rulesets" ([`frule`](@ref)s, [`rrule`](@ref)s,)
to avoid having to code all your own AD primitives / custom sensitives.

There are 3 main ways to access ChainRules rule sets in your AutoDiff system.

1. [Operator Overloading Generation](https://juliadiff.org/ChainRulesOverloadGeneration.jl/stable)
    - Use [ChainRulesOverloadGeneration.jl](https://github.com/JuliaDiff/ChainRulesOverloadGeneration.jl/).
    - This is primarily intended for operator overloading based AD systems which will generate overloads for primal functions based for their overloaded types based on the existence of an `rrule`/`frule`.
    - A source code generation based AD can also use this by overloading their transform generating function directly so as not to recursively generate a transform but to just return the rule.
    - This does not play nice with Revise.jl, adding or modifying rules in loaded files will not be reflected until a manual refresh, and deleting rules will not be reflected at all.
2. Source code transform based on inserting branches that check of `rrule`/`frule` return `nothing`
    - If the `rrule`/`frule` returns a rule result then use it, if it returns `nothing` then do normal AD path.
    - In theory type inference optimizes these branches out; in practice it may not.
    - This is a fairly simple Cassette overdub (or similar) of all calls, and is suitable for overloading based AD or source code transformation.
3. Source code transform based on `rrule`/`frule` method-table
    - If an applicable `rrule`/`frule` exists in the method table then use it, else generate normal AD path.
    - This avoids having branches in your generated code.
    - This requires maintaining your own back-edges.
    - This is pretty hardcore even by the standard of source code transformations.
