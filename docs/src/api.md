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

## Differentials
```@autodocs
Modules = [ChainRulesCore]
Pages = [
    "differentials/abstract_zero.jl",
    "differentials/one.jl",
    "differentials/composite.jl",
    "differentials/thunks.jl",
    "differentials/abstract_differential.jl",
    "differentials/notimplemented.jl",
]
Private = false
```

## Accumulation
```@docs
add!!
ChainRulesCore.is_inplaceable_destination
```

## Internal
```@docs
ChainRulesCore.AbstractTangent
ChainRulesCore.debug_mode
```

## Deprecated
```@docs
ChainRulesCore.extern
```
