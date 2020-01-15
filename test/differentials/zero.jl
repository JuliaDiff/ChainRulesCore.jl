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
    @test zero(@thunk(3)) === z
    @test zero(One()) === z
    @test zero(DoesNotExist()) === z
    @test zero(Composite{Tuple{Int,Int}}((1, 2))) === z

    # use mutable objects to test the strong `===` condition
    x = ones(2)
    @test muladd(Zero(), 2, x) === x
    @test muladd(2, Zero(), x) === x
    @test muladd(Zero(), Zero(), x) === x
    @test muladd(2, 2, Zero()) === 4
    @test muladd(x, Zero(), Zero()) === Zero()
    @test muladd(Zero(), x, Zero()) === Zero()
    @test muladd(Zero(), Zero(), Zero()) === Zero()
end
