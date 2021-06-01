@testset "NO_FIELDS" begin
    @test (@test_deprecated NO_FIELDS) isa NoTangent
end
