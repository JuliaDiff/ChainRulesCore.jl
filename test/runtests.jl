# TODO: more tests!

using ChainRulesCore, Test
using LinearAlgebra: Diagonal
using ChainRulesCore: extern, accumulate, accumulate!, store!, @scalar_rule,
    Wirtinger, wirtinger_primal, wirtinger_conjugate, add_wirtinger, mul_wirtinger,
    Zero, add_zero, mul_zero, One, add_one, mul_one, Casted, cast, add_casted, mul_casted,
    DNE, Thunk, Casted, DNERule, WirtingerRule
using Base.Broadcast: broadcastable

@testset "ChainRulesCore" begin
    include("differentials.jl")
    include("rules.jl")
end
