module ChainRulesCore
using Base.Broadcast: broadcasted, Broadcasted, broadcastable, materialize, materialize!
using Base.Meta
using Compat: hasfield

export frule, rrule  # core function
# rule configurations
export RuleConfig, HasReverseMode, NoReverseMode, HasForwardsMode, NoForwardsMode
export frule_via_ad, rrule_via_ad
# definition helper macros
export @non_differentiable, @opt_out, @scalar_rule, @thunk, @not_implemented
export ProjectTo, canonicalize, unthunk  # differential operations
export add!!  # gradient accumulation operations
# differentials
export Tangent, NoTangent, InplaceableThunk, Thunk, ZeroTangent, AbstractZero, AbstractThunk

include("compat.jl")
include("debug_mode.jl")

include("differentials/abstract_differential.jl")
include("differentials/abstract_zero.jl")
include("differentials/thunks.jl")
include("differentials/composite.jl")
include("differentials/notimplemented.jl")

include("accumulation.jl")
include("projection.jl")

include("config.jl")
include("rules.jl")
include("rule_definition_tools.jl")

include("deprecated.jl")

end # module
