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
]
Private = false
```

## Accumulation
```@docs
add!!
ChainRulesCore.is_inplaceable_destination
```

## Ruleset Loading
```@autodocs
Modules = [ChainRulesCore]
Pages = ["ruleset_loading.jl"]
Private = false
```

## Internal
```@docs
ChainRulesCore.AbstractDifferential
ChainRulesCore.debug_mode
ChainRulesCore.clear_new_rule_hooks!
```
