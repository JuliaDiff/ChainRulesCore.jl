@testset "One" begin
    o = One()
    @test extern(o) === true
    @test o + o == 2
    @test o + 1 == 2
    @test 1 + o == 2
    @test o * o == o
    @test o * 17 == 17
    @test 6 * o == 6
    @test dot(2 + im, o) == 2 - im
    @test dot(o, 2 + im) == 2 + im
    for x in o
        @test x === o
    end
    @test broadcastable(o) isa Ref{One}
    @test conj(o) == o
    
    @test reim(o) === (One(), Zero())
    @test real(o) === One()
    @test imag(o) === Zero()

    @test complex(o) === o
    @test complex(o, Zero()) === o
    @test complex(Zero(), o) === im
end
