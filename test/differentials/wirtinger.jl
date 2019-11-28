@testset "Wirtinger" begin
    w = Wirtinger(1+1im, 2+2im)
    @test wirtinger_primal(w) == 1+1im
    @test wirtinger_conjugate(w) == 2+2im
    @test w + w == Wirtinger(2+2im, 4+4im)

    @test w + One() == w + 1 == w + Thunk(()->1) == Wirtinger(2+1im, 2+2im)
    @test w * One() == One() * w == w
    @test w * 2 == 2 * w == Wirtinger(2 + 2im, 4 + 4im)

    # TODO: other + methods stack overflow
    @test_throws ErrorException w*w
    @test_throws ArgumentError extern(w)
    for x in w
        @test x === w
    end
    @test broadcastable(w) == w
    @test_throws MethodError conj(w)
end
