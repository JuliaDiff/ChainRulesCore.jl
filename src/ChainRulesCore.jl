module ChainRulesCore
using Base.Broadcast: broadcasted, Broadcasted, broadcastable, materialize, materialize!
using Base.Meta
using LinearAlgebra
using Compat: hasfield, hasproperty, ismutabletype

export frule, rrule  # core function
# rule configurations
export RuleConfig, HasReverseMode, NoReverseMode, HasForwardsMode, NoForwardsMode
export frule_via_ad, rrule_via_ad
# definition helper macros
export @non_differentiable, @opt_out, @scalar_rule, @thunk, @not_implemented
export ProjectTo, canonicalize, unthunk, zero_tangent  # tangent operations
export add!!, is_inplaceable_destination  # gradient accumulation operations
export ignore_derivatives, @ignore_derivatives
# tangents
export StructuralTangent, Tangent, MutableTangent, NoTangent, InplaceableThunk, Thunk, ZeroTangent, AbstractZero, AbstractThunk

include("debug_mode.jl")

include("tangent_types/abstract_tangent.jl")
include("tangent_types/structural_tangent.jl")
include("tangent_types/abstract_zero.jl")
include("tangent_types/thunks.jl")
include("tangent_types/notimplemented.jl")

include("tangent_arithmetic.jl")
include("accumulation.jl")
include("projection.jl")

include("config.jl")
include("rules.jl")
include("rule_definition_tools.jl")
include("ignore_derivatives.jl")

include("deprecated.jl")

# SparseArrays support on Julia < 1.9
if !isdefined(Base, :get_extension)
    include("../ext/ChainRulesCoreSparseArraysExt.jl")
end

end # module
