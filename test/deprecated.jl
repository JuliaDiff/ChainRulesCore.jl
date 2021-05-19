@testset "deprecations" begin
    @test ChainRulesCore.AbstractDifferential === ChainRulesCore.AbstractTangent
    @test Zero === ZeroTangent
    @test DoesNotExist === NoTangent
    @test Composite === Tangent
end
