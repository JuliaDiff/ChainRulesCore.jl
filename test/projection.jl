@testset "projection" begin
    @testset "Number types" begin
        @test 3.2 == project(1.0, 3.2)
        @test 3.2 == project(1.0, 3.2 + 3im)
        @test 3.2f0 == project(Float32, 1.0f0, 3.2 - 3im)

        @test 0.0 == project(1.1, ZeroTangent())
        @test 3.2 == project(1.0, @thunk(3.2))
    end
end
