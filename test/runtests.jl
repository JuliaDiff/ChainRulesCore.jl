using Base.Broadcast: broadcastable
using ChainRulesCore
using LinearAlgebra: Diagonal
using Test

@testset "ChainRulesCore" begin
    @testset "differentials" begin
        include("differentials/abstract_zero.jl")
        include("differentials/one.jl")
        include("differentials/thunks.jl")
        include("differentials/composite.jl")
    end

    include("rules.jl")
end
