using Base.Broadcast: broadcastable
using BenchmarkTools
using Preferences
using UUIDs

# Test Float32 value for int2float
if "INT2FLOAT" ∈ keys(ENV) && ENV["INT2FLOAT"] == "Float32"
    chainrulescore_uuid = UUID("d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4")
    set_preferences!(chainrulescore_uuid, "int2float" => "Float32")
end

using ChainRulesCore
using LinearAlgebra
using LinearAlgebra.BLAS: ger!, gemv!, gemv, scal!
using StaticArrays
using SparseArrays
using Test

int2float(x) = ProjectTo(1)(x)

@testset "ChainRulesCore" begin
    @testset "differentials" begin
        include("tangent_types/abstract_zero.jl")
        include("tangent_types/thunks.jl")
        include("tangent_types/tangent.jl")
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
