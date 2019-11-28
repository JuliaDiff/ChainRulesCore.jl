# TODO: more tests!
using Test
using ChainRulesCore
using LinearAlgebra: Diagonal
using ChainRulesCore: extern, accumulate, accumulate!, store!,
    Composite, @scalar_rule,
    Wirtinger, wirtinger_primal, wirtinger_conjugate,
    Zero, One, DoesNotExist, Thunk
using Base.Broadcast: broadcastable

@testset "ChainRulesCore" begin
    include("differentials_common.jl")
    @testset "differentials" begin
        include("differentials/wirtinger.jl")
        include("differentials/zero.jl")
        include("differentials/one.jl")
        include("differentials/thunks.jl")
        include("differentials/composite.jl")
    end

    include("rules.jl")
end
