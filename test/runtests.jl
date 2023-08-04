using Base.Broadcast: broadcastable
using BenchmarkTools
using ChainRulesCore
using LinearAlgebra
using LinearAlgebra.BLAS: ger!, gemv!, gemv, scal!
using StaticArrays
using SparseArrays
using Test

@testset "ChainRulesCore" begin
    @testset "differentials" begin
        include("tangent_types/abstract_zero.jl")
        include("tangent_types/thunks.jl")
        include("tangent_types/structural_tangent.jl")
        include("tangent_types/notimplemented.jl")
    end

    include("accumulation.jl")
    include("projection.jl")

    include("rules.jl")
    include("rule_definition_tools.jl")
    include("config.jl")
    include("ignore_derivatives.jl")

    include("deprecated.jl")
end
