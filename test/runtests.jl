# TODO: more tests!
using Test
using ChainRulesCore
using LinearAlgebra: Diagonal
using ChainRulesCore: extern, accumulate, accumulate!, store!, @scalar_rule,
    Wirtinger, wirtinger_primal, wirtinger_conjugate,
    Zero, One, Casted, cast,
    DNE, Thunk, Casted, DNERule, WirtingerRule
using Base.Broadcast: broadcastable

@testset "ChainRulesCore" begin
    include("differentials.jl")
    include("rules.jl")
    include("rule_types.jl")
end
