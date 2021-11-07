@testset "AbstractZero" begin
    @testset "iszero" begin
        @test iszero(ZeroTangent())
        @test iszero(NoTangent())
    end

    @testset "Linear operators" begin
        @test getindex(ZeroTangent(), 1) === ZeroTangent()
        @test getindex(NoTangent(), 1, 2) === NoTangent()

        @test view(ZeroTangent(), 1) == ZeroTangent()
        @test view(NoTangent(), 1, 2) == NoTangent()

        @test sum(ZeroTangent()) == ZeroTangent()
        @test sum(NoTangent(); dims=2) == NoTangent()

        @test reshape(NoTangent(), (1, :)) === NoTangent()
        @test reshape(ZeroTangent(), 2, 3, 4, 5) === ZeroTangent()

        @test reverse(NoTangent()) === NoTangent()
        @test reverse(ZeroTangent()) === ZeroTangent()
        @test reverse(ZeroTangent(); dims=1) === ZeroTangent()
        @test reverse(ZeroTangent(), 2, 5) === ZeroTangent()
    end

    @testset "ZeroTangent" begin
        z = ZeroTangent()
        @test z + z === z
        @test z + 1 === 1
        @test 1 + z === 1
        @test z - z === z
        @test z - 1 === -1
        @test 1 - z === 1
        @test -z === z
        @test z * z === z
        @test z * 11.1 === ZeroTangent()
        @test 12.3 * z === ZeroTangent()
        @test dot(z, z) === z
        @test dot(z, 1.8) === z
        @test dot(2.1, z) === z
        @test dot([1, 2], z) === z
        @test dot(z, [1, 2]) === z
        for x in z
            @test x === z
        end
        @test broadcastable(z) isa Ref{ZeroTangent}
        @test zero(@thunk(3)) === z
        @test zero(NoTangent()) === z
        @test zero(ZeroTangent) === z
        @test zero(NoTangent) === z
        @test zero(Tangent{Tuple{Int,Int}}((1, 2))) === z
        for f in (transpose, adjoint, conj)
            @test f(z) === z
        end
        @test z / 2 === z / [1, 3] === z

        @test eltype(z) === ZeroTangent
        @test eltype(ZeroTangent) === ZeroTangent

        # use mutable objects to test the strong `===` condition
        x = ones(2)
        @test muladd(ZeroTangent(), 2, x) === x
        @test muladd(2, ZeroTangent(), x) === x
        @test muladd(ZeroTangent(), ZeroTangent(), x) === x
        @test muladd(2, 2, ZeroTangent()) === 4
        @test muladd(x, ZeroTangent(), ZeroTangent()) === ZeroTangent()
        @test muladd(ZeroTangent(), x, ZeroTangent()) === ZeroTangent()
        @test muladd(ZeroTangent(), ZeroTangent(), ZeroTangent()) === ZeroTangent()

        @test reim(z) === (ZeroTangent(), ZeroTangent())
        @test real(z) === ZeroTangent()
        @test imag(z) === ZeroTangent()

        @test complex(z) === z
        @test complex(z, z) === z
        @test complex(z, 2.0) === Complex{Float64}(0.0, 2.0)
        @test complex(1.5, z) === Complex{Float64}(1.5, 0.0)
        @test Complex(z, 2.0) === Complex{Float64}(0.0, 2.0)
        @test Complex(1.5, z) === Complex{Float64}(1.5, 0.0)
        @test ComplexF64(z, 2.0) === Complex{Float64}(0.0, 2.0)
        @test ComplexF64(1.5, z) === Complex{Float64}(1.5, 0.0)

        @test convert(Bool, ZeroTangent()) === false
        @test convert(Int64, ZeroTangent()) === Int64(0)
        @test convert(Float32, ZeroTangent()) === 0.0f0
        @test convert(ComplexF64, ZeroTangent()) === 0.0 + 0.0im

        @test z[1] === z
        @test z[1:3] === z
        @test z[1, 2] === z
        @test getindex(z) === z
    end

    @testset "NoTangent" begin
        dne = NoTangent()
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

        @test ZeroTangent() + dne == dne
        @test dne + ZeroTangent() == dne
        @test ZeroTangent() - dne == dne
        @test dne - ZeroTangent() == dne

        @test ZeroTangent() * dne == ZeroTangent()
        @test dne * ZeroTangent() == ZeroTangent()
        @test dot(ZeroTangent(), dne) == ZeroTangent()
        @test dot(dne, ZeroTangent()) == ZeroTangent()

        for x in dne
            @test x === dne
        end
        @test broadcastable(dne) isa Ref{NoTangent}
        for f in (transpose, adjoint, conj)
            @test f(dne) === dne
        end
        @test dne / 2 === dne / [1, 3] === dne

        @test convert(Int64, NoTangent()) == 0
        @test convert(Float64, NoTangent()) == 0.0

        @test dne[1] === dne
        @test dne[1:3] === dne
        @test dne[1, 2] === dne
        @test getindex(dne) === dne
    end

    @testset "ambiguities" begin
        M = @eval module M
            using ChainRulesCore

            struct X{R,S} <: Number
                a::R
                b::S
                hasvalue::Bool

                function X{R,S}(a, b, hv=true) where {R,S}
                    isa(hv, Bool) || error("must be bool")
                    return new{R,S}(a, b, hv)
                end
            end
        end
        @test isempty(detect_ambiguities(M))
    end

    @testset "LinearAlgebra constructors" begin
        @test adjoint(ZeroTangent()) === ZeroTangent()
        @test transpose(ZeroTangent()) === ZeroTangent()
        @test Adjoint(ZeroTangent()) === ZeroTangent()
        @test Transpose(ZeroTangent()) === ZeroTangent()
        @test Symmetric(ZeroTangent()) === ZeroTangent()
        @test Hermitian(ZeroTangent(), :U) === ZeroTangent()

        # Multidiagonal
        @test Bidiagonal([1, 2, 3], ZeroTangent(), :U) == Diagonal(1:3)
        @test Bidiagonal(ZeroTangent(), 1:3, :U) == [0 1 0 0; 0 0 2 0; 0 0 0 3; 0 0 0 0]
        @test Tridiagonal(ZeroTangent(), ZeroTangent(), [4, 5]) == [0 4 0; 0 0 5; 0 0 0]
        @test Tridiagonal(ZeroTangent(), 1:3, ZeroTangent()) == Diagonal(1:3)
    end

    @testset "promote vectors" begin
        v64s = (Vector{Float64}, Vector{Float64})
        @test v64s == typeof.(ChainRulesCore._promote_vectors(fill(pi,3), [1,2,3]))
        @test v64s == typeof.(ChainRulesCore._promote_vectors(Any[1,2,pi], rand(2)))
        @test v64s == typeof.(ChainRulesCore._promote_vectors(randn(3) .> 1, rand(10)))

        @test length.(ChainRulesCore._promote_vectors(Any[1,2,pi], rand(2))) == (3, 2)
        @test length.(ChainRulesCore._promote_vectors(randn(3) .> 1, rand(10))) == (3, 10)

        _samet((x, y)) = typeof(x) == typeof(y)
        @test _samet(ChainRulesCore._promote_vectors(sparse([1,0,3]), [4,5,6]))
        @test _samet(ChainRulesCore._promote_vectors(SA[1/2, 3/4], [5, 6]))
        # The last isn't actually realistic, since all the constructors which take several
        # vectors demand both same type and different length, so cannot accept StaticArrays.
    end
end
