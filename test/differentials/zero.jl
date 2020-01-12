@testset "Zero" begin
    z = Zero()
    @test extern(z) === false
    @test z + z === z
    @test z + 1 === 1
    @test 1 + z === 1
    @test z * z === z
    @test z * 1 === 0
    @test 1 * z === 0
    for x in z
        @test x === z
    end
    @test broadcastable(z) isa Ref{Zero}
    @test conj(z) === z
end
