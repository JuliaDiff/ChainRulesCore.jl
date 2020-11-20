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
        @test z - z === z
        @test z - 1 === -1
        @test 1 - z === 1
        @test -z === z
        @test z * z === z
        @test z * 11.1 === Zero()
        @test 12.3 * z === Zero()
        @test dot(z, z) === z
        @test dot(z, 1.8) === z
        @test dot(2.1, z) === z
        @test dot([1, 2], z) === z
        @test dot(z, [1, 2]) === z
        for x in z
            @test x === z
        end
        @test broadcastable(z) isa Ref{Zero}
        @test zero(@thunk(3)) === z
        @test zero(One()) === z
        @test zero(DoesNotExist()) === z
        @test zero(One) === z
        @test zero(Zero) === z
        @test zero(DoesNotExist) === z
        @test zero(Composite{Tuple{Int,Int}}((1, 2))) === z
        for f in (transpose, adjoint, conj)
            @test f(z) === z
        end
        @test z / 2 === z / [1, 3] === z

        @test eltype(z) === Zero
        @test eltype(Zero) === Zero

        # use mutable objects to test the strong `===` condition
        x = ones(2)
        @test muladd(Zero(), 2, x) === x
        @test muladd(2, Zero(), x) === x
        @test muladd(Zero(), Zero(), x) === x
        @test muladd(2, 2, Zero()) === 4
        @test muladd(x, Zero(), Zero()) === Zero()
        @test muladd(Zero(), x, Zero()) === Zero()
        @test muladd(Zero(), Zero(), Zero()) === Zero()
        
        @test reim(z) === (Zero(), Zero())
        @test real(z) === Zero()
        @test imag(z) === Zero()

        @test complex(z) === z
        @test complex(z, z) === z
        @test complex(z, 2.0) === Complex{Float64}(0.0, 2.0)
        @test complex(1.5, z) === Complex{Float64}(1.5, 0.0)

        @test convert(Int64, Zero()) == 0
        @test convert(Float64, Zero()) == 0.0
    end

    @testset "DoesNotExist" begin
        dne = DoesNotExist()
        @test_throws Exception extern(dne)
        @test dne + dne == dne
        @test dne + 1 == 1
        @test 1 + dne == 1
        @test dne - dne == dne
        @test dne - 1 == -1
        @test 1 - dne == 1
        @test -dne == dne
        @test dne * dne == dne
        @test dne * 11.1 == dne
        @test 12.1 * dne == dne
        @test dot(dne, dne) == dne
        @test dot(dne, 17.2) == dne
        @test dot(11.9, dne) == dne

        @test Zero() + dne == dne
        @test dne + Zero() == dne
        @test Zero() - dne == dne
        @test dne - Zero() == dne

        @test Zero() * dne == Zero()
        @test dne * Zero() == Zero()
        @test dot(Zero(), dne) == Zero()
        @test dot(dne, Zero()) == Zero()

        for x in dne
            @test x === dne
        end
        @test broadcastable(dne) isa Ref{DoesNotExist}
        for f in (transpose, adjoint, conj)
            @test f(dne) === dne
        end
        @test dne / 2 === dne / [1, 3] === dne

        @test convert(Int64, DoesNotExist()) == 0
        @test convert(Float64, DoesNotExist()) == 0.0
    end
end
