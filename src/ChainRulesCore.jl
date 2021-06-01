module ChainRulesCore
using Base.Broadcast: broadcasted, Broadcasted, broadcastable, materialize, materialize!
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseVector, SparseMatrixCSC
using Compat: hasfield

export on_new_rule, refresh_rules  # generation tools
export frule, rrule  # core function
export @non_differentiable, @scalar_rule, @thunk, @not_implemented  # definition helper macros
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
include("differentials/notimplemented.jl")

include("differential_arithmetic.jl")
include("accumulation.jl")

include("rules.jl")
include("rule_definition_tools.jl")
include("ruleset_loading.jl")

include("deprecated.jl")
include("precompile.jl")

end # module
