module ChainRulesCore
using Base.Broadcast: broadcasted, Broadcasted, broadcastable, materialize, materialize!
using MuladdMacro: @muladd

export frule, rrule
export @frule, @rrule, @scalar_rule, @thunk
export canonicalize, extern, unthunk
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

end # module
