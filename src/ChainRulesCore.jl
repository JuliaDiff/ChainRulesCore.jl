module ChainRulesCore
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable

export frule, rrule
export wirtinger_conjugate, wirtinger_primal, refine_differential
export @scalar_rule, @thunk
export extern, store!
export Wirtinger, Zero, One, DoesNotExist, Thunk, InplaceableThunk
export NO_FIELDS

include("differentials.jl")
include("differential_arithmetic.jl")
include("operations.jl")
include("rules.jl")
include("rule_definition_tools.jl")

Base.@deprecate_binding DNE DoesNotExist

end # module
