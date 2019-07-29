# TODO: more tests!

using AbstractChainRules, Test
using LinearAlgebra: Diagonal
using AbstractChainRules: extern, accumulate, accumulate!, store!, @scalar_rule,
    Wirtinger, wirtinger_primal, wirtinger_conjugate, add_wirtinger, mul_wirtinger,
    Zero, add_zero, mul_zero, One, add_one, mul_one, Casted, cast, add_casted, mul_casted,
    DNE, Thunk, Casted, DNERule
using Base.Broadcast: broadcastable

#include("test_util.jl")

@testset "AbstractChainRules" begin
    include("differentials.jl")
    include("rules.jl")
end
