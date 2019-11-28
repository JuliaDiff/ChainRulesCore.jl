@testset "One" begin
    o = One()
    @test extern(o) === true
    @test o + o == 2
    @test o + 1 == 2
    @test 1 + o == 2
    @test o * o == o
    @test o * 1 == 1
    @test 1 * o == 1
    for x in o
        @test x === o
    end
    @test broadcastable(o) isa Ref{One}
    @test conj(o) == o
end
