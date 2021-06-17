using Base.Broadcast: broadcastable
using BenchmarkTools
using ChainRulesCore
using LinearAlgebra
using LinearAlgebra.BLAS
using StaticArrays
using SparseArrays
using Test

@testset "ChainRulesCore" begin
    @testset "differentials" begin
        include("differentials/abstract_zero.jl")
        include("differentials/thunks.jl")
        include("differentials/composite.jl")
        include("differentials/combinations.jl")
        include("differentials/notimplemented.jl")
    end

    include("accumulation.jl")

    include("rules.jl")
    include("rule_definition_tools.jl")
    include("config.jl")
end
