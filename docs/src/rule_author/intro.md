# Introduction

This section of the docs will tell you everything you need to know about writing rules for your package.

It will help with understanding tangent types, the anatomy of the `frule` and
the `rrule`, and provide tips on writing good rules, as well as how to test them easily
using finite differences.

This section also outlines some ChainRules superpowers that can be considered advanced usage.
Most users can ignore these.
However:
- If you are writing rules with abstractly typed arguments, read about [`ProjectTo`](@ref projectto).
- If you want to opt out of using the abstractly typed rule for certain argument types, read about [`@opt_out`](@ref opt_out).
- If you are writing rules for higher order functions, read about [calling back into AD](@ref config).
- If you want to accumulate gradients in-place to avoid extra allocations, read about [gradient accumulation](@ref grad_acc).
