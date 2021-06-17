module ChainRulesCore
using Base.Broadcast: broadcasted, Broadcasted, broadcastable, materialize, materialize!
using LinearAlgebra
using SparseArrays: SparseVector, SparseMatrixCSC
using Compat: hasfield

export frule, rrule  # core function
# rule configurations
export RuleConfig, HasReverseMode, NoReverseMode, HasForwardsMode, NoForwardsMode
export frule_via_ad, rrule_via_ad
# definition helper macros
export @non_differentiable, @scalar_rule, @thunk, @not_implemented
export canonicalize, extern, unthunk  # differential operations
export add!!  # gradient accumulation operations
# differentials
export Tangent, NoTangent, InplaceableThunk, Thunk, ZeroTangent, AbstractZero, AbstractThunk

include("compat.jl")
include("debug_mode.jl")

include("differentials/abstract_differential.jl")
include("differentials/abstract_zero.jl")
include("differentials/thunks.jl")
include("differentials/composite.jl")
include("differentials/combinations.jl")
include("differentials/notimplemented.jl")

include("differential_arithmetic.jl")
include("accumulation.jl")

include("config.jl")
include("rules.jl")
include("rule_definition_tools.jl")

include("deprecated.jl")

end # module
