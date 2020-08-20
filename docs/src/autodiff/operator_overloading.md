# Operator Overloading

The principle interface for using the operator overload generation method is [`on_new_rule`](@ref).
This function allows one to register a hook to be run every time a new rule is defined.
The hook receives a signature type-type as input, and generally will use `eval` to define
and overload of AD systems overloaded type.
For example, using the signature type `Tuple{typeof(+), Real, Real}` to define 
`+(::DualNumber, ::DualNumber)` as calling the `frule` for `+`.
A signature type tuple always has the form:
`Tuple{typeof(operation), typeof{pos_arg1}, typeof{pos_arg2...}}`, where `pos_arg1` is the
first positional argument.
One can dispatch on the signature type, to make rules with argument types your AD does not support not call `eval`;
or more simply you can just use conditions for this.

## Examples

### ForwardDiffZero
The overload generation hook in this example is: `define_dual_overload`.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read(joinpath(@__DIR__,"../../../test/demos/forwarddiffzero.jl"), String))
```
""")
````

### ReverseDiffZero
The overload generation hook in this example is: `define_tracked_overload`.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read(joinpath(@__DIR__,"../../../test/demos/reversediffzero.jl"), String))
```
""")
````

