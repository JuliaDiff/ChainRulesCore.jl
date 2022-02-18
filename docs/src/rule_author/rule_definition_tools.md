# [Using rule definition tools](@id ruletools)

Rule definition tools can help you write more `frule`s and the `rrule`s with less lines of code.

## [`@non_differentiable`](@ref)

For non-differentiable functions the [`@non_differentiable`](@ref) macro can be used.
For example, instead of manually defining the `frule` and the `rrule` for string concatenation `*(String..)`, the macro call
```julia
@non_differentiable *(String...)
```
defines the following `frule` and `rrule` automatically
```julia
function ChainRulesCore.frule(var"##_#1600", ::Core.Typeof(*), String::Any...; kwargs...)
    return (*(String...; kwargs...), NoTangent())
end
function ChainRulesCore.rrule(::Core.Typeof(*), String::Any...; kwargs...)
    return (*(String...; kwargs...), function var"*_pullback"(_)
        (ZeroTangent(), ntuple((_->NoTangent()), 0 + length(String))...)
    end)
end
```
Note that the types of arguments are propagated to the `frule` and `rrule` definitions.
This is needed in case the function differentiable for some but not for other types of arguments.
For example `*(1, 2, 3)` is differentiable, and is not defined with the macro call above.

## [`@scalar_rule`](@ref)

For functions involving only scalars, i.e. subtypes of `Number` (no `struct`s, `String`s...), both the `frule` and the `rrule` can be defined using a single [`@scalar_rule`](@ref) macro call.

Note that the function does not have to be $\mathbb{R} \rightarrow \mathbb{R}$.
In fact, any number of scalar arguments is supported, as is returning a tuple of scalars.

See docstrings for the comprehensive usage instructions.
