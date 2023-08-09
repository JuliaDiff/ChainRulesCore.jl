using Base.Broadcast: broadcastable
using BenchmarkTools
using Preferences
using UUIDs

# Test Float32 value for int2float

if "INT2FLOAT" ∈ keys(ENV)
    env_int2float = ENV["INT2FLOAT"]
end

if env_int2float ∈ ["Float32", "Float16"]
    chainrulescore_uuid = UUID("d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4")
    set_preferences!(chainrulescore_uuid, "int2float" => env_int2float)
    println("")
    println("Running ChainRulesCore tests with $env_int2float")
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
