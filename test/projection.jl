struct Fred
    a::Float64
end

Base.zero(::Fred) = Fred(0.0)
Base.zero(::Type{Fred}) = Fred(0.0)

@testset "projection" begin
    @testset "fallback" begin
        @test Fred(1.2) == project(Fred, Fred(1.2))
        @test Fred(0.0) == project(Fred, ZeroTangent())
        @test Fred(3.2) == project(Fred, @thunk(Fred(3.2)))
    end

    @testset "to Real" begin
        # Float64
        @test 3.2 == project(Float64, 3.2)
        @test 0.0 == project(Float64, ZeroTangent())
        @test 3.2 == project(Float64, @thunk(3.2))

        # down
        @test 3.2 == project(Float64, 3.2 + 3im)
        @test 3.2f0 == project(Float32, 3.2)
        @test 3.2f0 == project(Float32, 3.2 - 3im)

        # up
        @test 2.0 == project(Float64, 2.0f0)
    end

    @testset "to Number" begin
        # Complex
        @test 2.0 + 0.0im == project(ComplexF64, 2.0 + 0.0im)

        # down
        @test 2.0 + 0.0im == project(ComplexF64, 2.0)
        @test 0.0 + 0.0im == project(ComplexF64, ZeroTangent())
        @test 0.0 + 0.0im == project(ComplexF64, @thunk(ZeroTangent()))

        # up
        @test 2.0 + 0.0im == project(ComplexF64, 2.0)
    end

    @testset "to Array" begin
        # to an array of numbers
        x = zeros(2, 2)
        @test [1.0 2.0; 3.0 4.0] == project(typeof(x), [1.0 2.0; 3.0 4.0]; info=preproject(x))
        @test x == project(typeof(x), ZeroTangent(); info=preproject(x))

        x = zeros(2)
        @test x == project(typeof(x), @thunk(ZeroTangent()); info=preproject(x))

        x = zeros(Float32, 2, 2)
        @test x == project(typeof(x), [0.0 0; 0 0]; info=preproject(x))

        x = [1.0 0; 0 4]
        @test x == project(typeof(x), Diagonal([1.0, 4]); info=preproject(x))

        # to a array of structs
        x = [Fred(0.0), Fred(0.0)]
        @test x == project(typeof(x), [Fred(0.0), Fred(0.0)]; info=preproject(x))
        @test x == project(typeof(x), [ZeroTangent(), ZeroTangent()]; info=preproject(x))
        @test x == project(typeof(x), [ZeroTangent(), @thunk(Fred(0.0))]; info=preproject(x))
        @test x == project(typeof(x), ZeroTangent(); info=preproject(x))
        @test x == project(typeof(x), @thunk(ZeroTangent()); info=preproject(x))

        x = [Fred(1.0) Fred(0.0); Fred(0.0) Fred(4.0)]
        @test x == project(typeof(x), Diagonal([Fred(1.0), Fred(4.0)]); info=preproject(x))
    end

    @testset "to Diagonal" begin
        d_F64 = Diagonal([0.0, 0.0])
        d_F32 = Diagonal([0.0f0, 0.0f0])
        d_C64 = Diagonal([0.0 + 0im, 0.0])
        d_Fred = Diagonal([Fred(0.0), Fred(0.0)])

        # from Matrix
        @test d_F64 == project(typeof(d_F64), zeros(2, 2); info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), zeros(Float32, 2, 2); info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), zeros(ComplexF64, 2, 2); info=preproject(d_F64))

        # from Diagonal of Numbers
        @test d_F64 == project(typeof(d_F64), d_F64; info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), d_F32; info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), d_C64; info=preproject(d_F64))

        # from Diagonal of AbstractTangent
        @test d_F64 == project(typeof(d_F64), ZeroTangent(); info=preproject(d_F64))
        @test d_C64 == project(typeof(d_C64), ZeroTangent(); info=preproject(d_C64))
        @test d_F64 == project(typeof(d_F64), @thunk(ZeroTangent()); info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), Diagonal([ZeroTangent(), ZeroTangent()]); info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), Diagonal([ZeroTangent(), @thunk(ZeroTangent())]); info=preproject(d_F64))

        # from Diagonal of structs
        @test d_Fred == project(typeof(d_Fred), ZeroTangent(); info=preproject(d_Fred))
        @test d_Fred == project(typeof(d_Fred), @thunk(ZeroTangent()); info=preproject(d_Fred))
        @test d_Fred == project(typeof(d_Fred), Diagonal([ZeroTangent(), ZeroTangent()]); info=preproject(d_Fred))

        # from Tangent
        @test d_F64 == project(typeof(d_F64), Tangent{Diagonal}(;diag=[0.0, 0.0]); info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), Tangent{Diagonal}(;diag=[0.0f0, 0.0f0]); info=preproject(d_F64))
        @test d_F64 == project(typeof(d_F64), Tangent{Diagonal}(;diag=[ZeroTangent(), @thunk(ZeroTangent())]); info=preproject(d_F64))
    end

    @testset "to Symmetric" begin
        data = [1.0 2; 3 4]

        x = Symmetric(data)
        @test x == project(typeof(x), data; info=preproject(x))

        x = Symmetric(data, :L)
        @test x == project(typeof(x), data; info=preproject(x))

        data = [1.0 0; 0 4]
        x = Symmetric(data)
        @test x == project(typeof(x), Diagonal([1.0, 4.0]); info=preproject(x))

        data = [0.0 0; 0 0]
        x = Symmetric(data)
        @test x == project(typeof(x), ZeroTangent(); info=preproject(x))
        @test x == project(typeof(x), @thunk(ZeroTangent()); info=preproject(x))
    end
end
