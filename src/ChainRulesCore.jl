module ChainRulesCore
using Base.Broadcast: broadcasted, Broadcasted, broadcastable, materialize, materialize!
using MuladdMacro: @muladd

export on_new_rule, refresh_rules, clear_new_rule_hooks!  # generation tools
export frule, rrule  # core function
export @scalar_rule, @thunk  # defination helper macros
export canonicalize, extern, unthunk  # differential operations
# differentials
export Composite, DoesNotExist, InplaceableThunk, One, Thunk, Zero, AbstractZero, AbstractThunk
export NO_FIELDS

include("compat.jl")
include("debug_mode.jl")

include("differentials/abstract_differential.jl")
include("differentials/abstract_zero.jl")
include("differentials/one.jl")
include("differentials/thunks.jl")
include("differentials/composite.jl")

include("differential_arithmetic.jl")

include("rules.jl")
include("rule_definition_tools.jl")
include("ruleset_loading.jl")

end # module