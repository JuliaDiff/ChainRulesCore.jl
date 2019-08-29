module ChainRulesCore
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable

export AbstractRule, Rule, frule, rrule
export @scalar_rule, @thunk
export extern, cast, store!, Wirtinger, Zero, One, Casted, DNE, Thunk, DNERule

include("differentials.jl")
include("differential_arithmetic.jl")
include("rule_types.jl")
include("rules.jl")
include("rule_definition_tools.jl")
end # module
