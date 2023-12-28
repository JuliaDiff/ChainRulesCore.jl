# API Documentation

## Rules
```@autodocs
Modules = [ChainRulesCore]
Pages = ["rules.jl"]
Private = false
```

## Rule Definition Tools
```@autodocs
Modules = [ChainRulesCore]
Pages = ["rule_definition_tools.jl"]
Private = false
```

## Tangent Types
```@autodocs
Modules = [ChainRulesCore]
Pages = [
    "tangent_types/abstract_zero.jl",
    "tangent_types/one.jl",
    "tangent_types/structural_tangent.jl",
    "tangent_types/thunks.jl",
    "tangent_types/abstract_tangent.jl",
    "tangent_types/notimplemented.jl",
]
Private = false
```

## Accumulation
```@docs
add!!
ChainRulesCore.is_inplaceable_destination
```

## RuleConfig
```@autodocs
Modules = [ChainRulesCore]
Pages = ["config.jl"]
Private = false
```

## ProjectTo
```@docs
ProjectTo
```

## Ignoring gradients
```@docs
ignore_derivatives
@ignore_derivatives
```

## Internal
```@docs
ChainRulesCore.AbstractTangent
ChainRulesCore.debug_mode
ChainRulesCore.no_rrule
ChainRulesCore.no_frule
```
