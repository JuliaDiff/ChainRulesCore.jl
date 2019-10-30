module ChainRulesCore
using Base.Broadcast: broadcasted, Broadcasted, broadcastable, materialize, materialize!

export frule, rrule
export refine_differential, wirtinger_conjugate, wirtinger_primal
export @scalar_rule, @thunk
export extern, store!, unthunk
export Composite, DoesNotExist, InplaceableThunk, One, Thunk, Wirtinger, Zero
export NO_FIELDS

include("differentials.jl")
include("composite_core.jl")
include("differential_arithmetic.jl")

include("operations.jl")
include("rules.jl")
include("rule_definition_tools.jl")

Base.@deprecate_binding DNE DoesNotExist

end # module
