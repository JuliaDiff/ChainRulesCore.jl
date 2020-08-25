# Operator Overloading

The principal interface for using the operator overload generation method is [`on_new_rule`](@ref).
This function allows one to register a hook to be run every time a new rule is defined.
The hook receives a signature type-type as input, and generally will use `eval` to define
an overload of an AD system's overloaded type.
For example, using the signature type `Tuple{typeof(+), Real, Real}` to make 
`+(::DualNumber, ::DualNumber)` call the `frule` for `+`.
A signature type tuple always has the form:
`Tuple{typeof(operation), typeof{pos_arg1}, typeof{pos_arg2}, ...}`, where `pos_arg1` is the
first positional argument.
One can dispatch on the signature type to make rules with argument types your AD does not support not call `eval`;
or more simply you can just use conditions for this.
For example if your AD only supports `AbstractMatrix{Float64}` and `Float64` inputs you might write:
```julia
const ACCEPT_TYPE = Union{Float64, AbstractMatrix{Float64}} 
function define_overload(sig::Type{<:Tuple{F, Vararg{ACCEPT_TYPE}}) where F
    @eval quote
        # ...
    end
end
define_overload(::Any) = nothing  # don't do anything for any other signature

on_new_rule(frule, define_overload)
```

or you might write:
```julia
const ACCEPT_TYPES = (Float64, AbstractMatrix{Float64})
function define_overload(sig)
    sig = Base.unwrap_unionall(sig)  # not really handling most UnionAll,
    opT, argTs = Iterators.peel(sig.parameters)
    all(any(acceptT<: argT for acceptT in ACCEPT_TYPES) for argT in argTs) || return
    @eval quote
        # ...
    end
end

on_new_rule(frule, define_overload)
```

The generation of overloaded code is the responsibility of the AD implementor.
Packages like [ExprTools.jl](https://github.com/invenia/ExprTools.jl) can be helpful for this.
Its generally fairly simple, though can become complex if you need to handle complicated type-constraints.
Examples are shown below.

The hook is automatically triggered whenever a package is loaded.
It can also be triggers manually using `refresh_rules`(@ref).
This is useful for example if new rules are define in the REPL, or if a package defining rules is modified.
(Revise.jl will not automatically trigger).
When the rules are refreshed (automatically or manually), the hooks are only triggered on new/modified rules; not ones that have already had the hooks triggered on.

`clear_new_rule_hooks!`(@ref) clears all registered hooks.
It is useful to undo [`on_new_rule`] hook registration if you are iteratively developing your overload generation function.

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
