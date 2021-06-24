struct Fred
    a::Float64
end

Base.zero(::Fred) = Fred(0.0)
Base.zero(::Type{Fred}) = Fred(0.0)

@testset "projection" begin
    @testset "fallback" begin
        @test Fred(1.2) == projector(Fred(3.2))(Fred(1.2))
        @test Fred(0.0) == projector(Fred(3.2))(ZeroTangent())
        @test Fred(3.2) == projector(Fred(-0.2))(@thunk(Fred(3.2)))
    end

    @testset "to Real" begin
        # Float64
        @test 3.2 == projector(1.0)(3.2)
        @test 0.0 == projector(1.1)(ZeroTangent())
        @test 3.2 == projector(1.0)(@thunk(3.2))

        # down
        @test 3.2 == projector(1.0)(3.2 + 3im)
        @test 3.2f0 == projector(1.0f0)(3.2)
        @test 3.2f0 == projector(1.0f0)(3.2 - 3im)

        # up
        @test 2.0 == projector(2.0)(2.0f0)
    end

    @testset "to Number" begin
        # Complex
        @test 2.0 + 0.0im == projector(1.0im)(2.0 + 0.0im)

        # down
        @test 2.0 + 0.0im == projector(1.0im)(2.0)
        @test 0.0 + 0.0im == projector(1.0im)(ZeroTangent())
        @test 0.0 + 0.0im == projector(1.0im)(@thunk(ZeroTangent()))

        # up
        @test 2.0 + 0.0im == projector(2.0 + 1.0im)(2.0)
    end

    @testset "to Array" begin
        # to an array of numbers
        @test [1.0 2.0; 3.0 4.0] == projector(zeros(2, 2))([1.0 2.0; 3.0 4.0])
        @test zeros(2, 2) == projector([1.0 2; 3 4])(ZeroTangent())
        @test zeros(2) == projector([1.0, 2.0])(@thunk(ZeroTangent()))
        @test [1.0f0 2; 3 4] == projector(zeros(Float32, 2, 2))([1.0 2; 3 4])
        @test [1.0 0; 0 4] == projector(zeros(2, 2))(Diagonal([1.0, 4]))

        # to a array of structs
        @test [Fred(0.0), Fred(0.0)] == projector([Fred(0.0), Fred(0.0)])([Fred(0.0), Fred(0.0)])
        @test [Fred(0.0), Fred(0.0)] == projector([Fred(0.0), Fred(0.0)])([ZeroTangent(), ZeroTangent()])
        @test [Fred(0.0), Fred(3.2)] == projector([Fred(0.0), Fred(0.0)])([ZeroTangent(), @thunk(Fred(3.2))])
        @test [Fred(0.0), Fred(0.0)] == projector([Fred(1.0), Fred(2.0)])(ZeroTangent())
        @test [Fred(0.0), Fred(0.0)] == projector([Fred(0.0), Fred(0.0)])(@thunk(ZeroTangent()))
        diagfreds = [Fred(1.0) Fred(0.0); Fred(0.0) Fred(4.0)]
        @test diagfreds == projector(diagfreds)(Diagonal([Fred(1.0), Fred(4.0)]))
    end

    @testset "to Tangent" begin
        @test Tangent{Fred}(; a = 3.2,) == projector(Tangent, Fred(3.2))(Fred(3.2))
        @test Tangent{Fred}(; a = ZeroTangent(),) == projector(Tangent, Fred(3.2))(ZeroTangent())
        @test Tangent{Fred}(; a = ZeroTangent(),) == projector(Tangent, Fred(3.2))(@thunk(ZeroTangent()))

        @test projector(Tangent, Diagonal(zeros(2)))(Diagonal([1.0f0, 2.0f0])) isa Tangent
        @test projector(Tangent, Diagonal(zeros(2)))(ZeroTangent()) isa Tangent
        @test projector(Tangent, Diagonal(zeros(2)))(@thunk(ZeroTangent())) isa Tangent
    end

    @testset "to Diagonal" begin
        d_F64 = Diagonal([0.0, 0.0])
        d_F32 = Diagonal([0.0f0, 0.0f0])
        d_C64 = Diagonal([0.0 + 0im, 0.0])
        d_Fred = Diagonal([Fred(0.0), Fred(0.0)])

        # from Matrix
        @test d_F64 == projector(d_F64)(zeros(2, 2))
        @test d_F64 == projector(d_F64)(zeros(Float32, 2, 2))
        @test d_F64 == projector(d_F64)(zeros(ComplexF64, 2, 2))

        # from Diagonal of Numbers
        @test d_F64 == projector(d_F64)(d_F64)
        @test d_F64 == projector(d_F64)(d_F32)
        @test d_F64 == projector(d_F64)(d_C64)

        # from Diagonal of AbstractTangent
        @test d_F64 == projector(d_F64)(ZeroTangent())
        @test d_C64 == projector(d_C64)(ZeroTangent())
        @test d_F64 == projector(d_F64)(@thunk(ZeroTangent()))
        @test d_F64 == projector(d_F64)(Diagonal([ZeroTangent(), ZeroTangent()]))
        @test d_F64 == projector(d_F64)(Diagonal([ZeroTangent(), @thunk(ZeroTangent())]))

        # from Diagonal of structs
        @test d_Fred == projector(d_Fred)(ZeroTangent())
        @test d_Fred == projector(d_Fred)(@thunk(ZeroTangent()))
        @test d_Fred == projector(d_Fred)(Diagonal([ZeroTangent(), ZeroTangent()]))

        # from Tangent
        @test d_F64 == projector(d_F64)(Tangent{Diagonal}(;diag=[0.0, 0.0]))
        @test d_F64 == projector(d_F64)(Tangent{Diagonal}(;diag=[0.0f0, 0.0f0]))
        @test d_F64 == projector(d_F64)(Tangent{Diagonal}(;diag=[ZeroTangent(), @thunk(ZeroTangent())]))
    end

    @testset "to Symmetric" begin
        data = [1.0 2; 3 4]
        @test Symmetric(data) == projector(Symmetric(data))(data)
        @test Symmetric(data, :L) == projector(Symmetric(data, :L))(data)
        @test Symmetric(Diagonal(data)) == projector(Symmetric(data))(Diagonal(diag(data)))

        @test Symmetric(zeros(2, 2)) == projector(Symmetric(data))(ZeroTangent())
        @test Symmetric(zeros(2, 2)) == projector(Symmetric(data))(@thunk(ZeroTangent()))
    end
end
