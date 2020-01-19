@testset "AbstractZero" begin
    @testset "iszero" begin
        @test iszero(Zero())
        @test iszero(DoesNotExist())
    end

    @testset "Zero" begin
        z = Zero()
        @test extern(z) === false
        @test z + z === z
        @test z + 1 === 1
        @test 1 + z === 1
        @test z * z === z
        @test z * 1 === Zero()
        @test 1 * z === Zero()
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

    @testset "DoesNotExist" begin
        dne = DoesNotExist()
        @test_throws Exception extern(dne)
        @test dne + dne == dne
        @test dne + 1 == 1
        @test 1 + dne == 1
        @test dne * dne == dne
        @test dne * 1 == dne
        @test 1 * dne == dne

        @test Zero() + dne == dne
        @test dne + Zero() == dne

        @test Zero() * dne == Zero()
        @test dne * Zero() == Zero()

        for x in dne
            @test x === dne
        end
        @test broadcastable(dne) isa Ref{DoesNotExist}
        @test conj(dne) == dne
    end
end
