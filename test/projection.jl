struct Fred
    a::Float64
end
Base.zero(::Fred) = Fred(0.0)
Base.zero(::Type{Fred}) = Fred(0.0)

struct Freddy{T, N}
    a::Array{T, N}
end
Base.:(==)(a::Freddy, b::Freddy) = a.a == b.a

struct Mary
    a::Fred
end
#Base.zero(::Mary) = Mary(zero(Fred))
#Base.zero(::Type{Mary}) = Mary(zero(Fred))

@testset "projection" begin
    @testset "display" begin
        @test startswith(repr(ProjectTo(Fred(1.1))), "ProjectTo{Fred}(")
        @test repr(ProjectTo(1.1)) == "ProjectTo{Float64}()"
    end

    @testset "fallback" begin
        @test Fred(1.2) == ProjectTo(Fred(1.1))(Fred(1.2))
        @test Fred(0.0) == ProjectTo(Fred(1.1))(ZeroTangent())
        @test Fred(3.2) == ProjectTo(Fred(1.1))(@thunk(Fred(3.2)))
        @test Fred(1.2) == ProjectTo(Fred(1.1))(Tangent{Fred}(;a=1.2))

        # struct with complicated field
        x = Freddy(zeros(2,2))
        dx = Tangent{Freddy}(; a=ZeroTangent())
        @test x == ProjectTo(x)(dx)

        # nested structs
        f = Fred(0.0)
        tf = Tangent{Fred}(;a=ZeroTangent())
        m = Mary(f)
        dm = Tangent{Mary}(;a=tf)
        @test m == ProjectTo(m)(dm)
    end

    @testset "to Real" begin
        # to type as shorthand for passing primal
        @test ProjectTo(Float64) == ProjectTo(1.2)

        # Float64
        @test 3.2 == ProjectTo(Float64)(3.2)
        @test 0.0 == ProjectTo(Float64)(ZeroTangent())
        @test 3.2 == ProjectTo(Float64)(@thunk(3.2))

        # down
        @test 3.2 == ProjectTo(Float64)(3.2 + 3im)
        @test 3.2f0 == ProjectTo(Float32)(3.2)
        @test 3.2f0 == ProjectTo(Float32)(3.2 - 3im)

        # up
        @test 2.0 == ProjectTo(Float64)(2.0f0)
    end

    @testset "to Number" begin
        # To type, as short-hand for passing primal
        @test ProjectTo(ComplexF64) == ProjectTo(1.0 + 2.0im)

        # Complex
        @test 2.0 + 4.0im == ProjectTo(ComplexF64)(2.0 + 4.0im)

        # down
        @test 2.0 + 0.0im == ProjectTo(ComplexF64)(2.0)
        @test 0.0 + 0.0im == ProjectTo(ComplexF64)(ZeroTangent())
        @test 0.0 + 0.0im == ProjectTo(ComplexF64)(@thunk(ZeroTangent()))

        # up
        @test 2.0 + 0.0im == ProjectTo(ComplexF64)(2.0)
    end

    @testset "to Array" begin
        # to an array of numbers
        x = zeros(2, 2)
        @test [1.0 2.0; 3.0 4.0] == ProjectTo(x)([1.0 2.0; 3.0 4.0])
        @test x == ProjectTo(x)(ZeroTangent())

        x = zeros(2)
        @test x == ProjectTo(x)(@thunk(ZeroTangent()))

        x = zeros(Float32, 2, 2)
        @test x == ProjectTo(x)([0.0 0; 0 0])

        x = [1.0 0; 0 4]
        @test x == ProjectTo(x)(Diagonal([1.0, 4]))

        # to a array of structs
        x = [Fred(0.0), Fred(0.0)]
        @test x == ProjectTo(x)([Fred(0.0), Fred(0.0)])
        @test x == ProjectTo(x)([ZeroTangent(), ZeroTangent()])
        @test x == ProjectTo(x)([ZeroTangent(), @thunk(Fred(0.0))])
        @test x == ProjectTo(x)(ZeroTangent())
        @test x == ProjectTo(x)(@thunk(ZeroTangent()))

        x = [Fred(1.0) Fred(0.0); Fred(0.0) Fred(4.0)]
        @test x == ProjectTo(x)(Diagonal([Fred(1.0), Fred(4.0)]))
    end

    @testset "to Diagonal" begin
        d_F64 = Diagonal([0.0, 0.0])
        d_F32 = Diagonal([0.0f0, 0.0f0])
        d_C64 = Diagonal([0.0 + 0im, 0.0])
        d_Fred = Diagonal([Fred(0.0), Fred(0.0)])

        # from Matrix
        @test d_F64 == ProjectTo(d_F64)(zeros(2, 2))
        @test d_F64 == ProjectTo(d_F64)(zeros(Float32, 2, 2))
        @test d_F64 == ProjectTo(d_F64)(zeros(ComplexF64, 2, 2))

        # from Diagonal of Numbers
        @test d_F64 == ProjectTo(d_F64)(d_F64)
        @test d_F64 == ProjectTo(d_F64)(d_F32)
        @test d_F64 == ProjectTo(d_F64)(d_C64)

        # from Diagonal of AbstractTangent
        @test d_F64 == ProjectTo(d_F64)(ZeroTangent())
        @test d_C64 == ProjectTo(d_C64)(ZeroTangent())
        @test d_F64 == ProjectTo(d_F64)(@thunk(ZeroTangent()))
        @test d_F64 == ProjectTo(d_F64)(Diagonal([ZeroTangent(), ZeroTangent()]))
        @test d_F64 == ProjectTo(d_F64)(Diagonal([ZeroTangent(), @thunk(ZeroTangent())]))

        # from Diagonal of structs
        @test d_Fred == ProjectTo(d_Fred)(ZeroTangent())
        @test d_Fred == ProjectTo(d_Fred)(@thunk(ZeroTangent()))
        @test d_Fred == ProjectTo(d_Fred)(Diagonal([ZeroTangent(), ZeroTangent()]))

        # from Tangent
        @test d_F64 == ProjectTo(d_F64)(Tangent{Diagonal}(;diag=[0.0, 0.0]))
        @test d_F64 == ProjectTo(d_F64)(Tangent{Diagonal}(;diag=[0.0f0, 0.0f0]))
        @test d_F64 == ProjectTo(d_F64)(Tangent{Diagonal}(;diag=[ZeroTangent(), @thunk(ZeroTangent())]))
    end

    @testset "to Symmetric" begin
        data = [1.0 2; 3 4]

        x = Symmetric(data)
        @test x == ProjectTo(x)(data)

        x = Symmetric(data, :L)
        @test x == ProjectTo(x)(data)

        data = [1.0 0; 0 4]
        x = Symmetric(data)
        @test x == ProjectTo(x)(Diagonal([1.0, 4.0]))

        data = [0.0 0; 0 0]
        x = Symmetric(data)
        @test x == ProjectTo(x)(ZeroTangent())
        @test x == ProjectTo(x)(@thunk(ZeroTangent()))
    end
end
