@testset "combinations.jl" begin
    @test ZeroTangent() == setindex!(ZeroTangent(), @thunk(1.0), 1)
end
