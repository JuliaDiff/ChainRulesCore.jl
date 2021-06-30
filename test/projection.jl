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

struct TwoFields
    a::Float64
    c::Float64
end

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

        # two fields
        tf = TwoFields(3.0, 0.0)
        @test tf == ProjectTo(tf)(Tangent{TwoFields}(; a=3.0))
    end

    @testset "to Real" begin
        # Float64
        @test 3.2 == ProjectTo(1.0)(3.2)
        @test 0.0 == ProjectTo(1.0)(ZeroTangent())
        @test 3.2 == ProjectTo(1.0)(@thunk(3.2))

        # down
        @test 3.2 == ProjectTo(1.0)(3.2 + 3im)
        @test 3.2f0 == ProjectTo(1.0f0)(3.2)
        @test 3.2f0 == ProjectTo(1.0f0)(3.2 - 3im)

        # up
        @test 2.0 == ProjectTo(1.0)(2.0f0)
    end

    @testset "to Number" begin
        # Complex
        @test 2.0 + 4.0im == ProjectTo(1.0im)(2.0 + 4.0im)

        # down
        @test 2.0 + 0.0im == ProjectTo(1.0im)(2.0)
        @test 0.0 + 0.0im == ProjectTo(1.0im)(ZeroTangent())
        @test 0.0 + 0.0im == ProjectTo(1.0im)(@thunk(ZeroTangent()))

        # up
        @test 2.0 + 0.0im == ProjectTo(1.0im)(2.0)
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

    @testset "To Array of Arrays" begin
        # inner arrays have same type but different sizes
        x = [[1.0, 2.0, 3.0], [4.0, 5.0]]
        @test x == ProjectTo(x)(x)
        @test x == ProjectTo(x)([[1.0 + 2im, 2.0, 3.0], [4.0 + 2im, 5.0]])

        # This makes sure we don't fall for https://github.com/JuliaLang/julia/issues/38064
        @test [[0.0, 0.0, 0.0], [0.0, 0.0]] == ProjectTo(x)(ZeroTangent())
    end

    @testset "Array{Any} with really messy contents" begin
        # inner arrays have same type but different sizes
        x = [[1.0, 2.0, 3.0], [4.0+im 5.0], [[[Fred(1)]]]]
        @test x == ProjectTo(x)(x)
        @test x == ProjectTo(x)([[1.0+im, 2.0, 3.0], [4.0+im 5.0], [[[Fred(1)]]]])
        # using a different type for the 2nd element (Adjoint)
        @test x == ProjectTo(x)([[1.0+im, 2.0, 3.0], [4.0-im, 5.0]', [[[Fred(1)]]]])

        @test [[0.0, 0.0, 0.0], [0.0im 0.0], [[[Fred(0)]]]] == ProjectTo(x)(ZeroTangent())
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
        @test x == ProjectTo(x)(Tangent{typeof(x)}(; data=data, uplo=NoTangent()))

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

    @testset "to Transpose" begin
        x = rand(3, 4)
        t = transpose(x)
        mt = collect(t)
        a = adjoint(x)
        ma = collect(a)

        @test t == ProjectTo(t)(mt)
        @test t == ProjectTo(t)(ma)
        @test zeros(4, 3) == ProjectTo(t)(ZeroTangent())
        @test zeros(4, 3) == ProjectTo(t)(Tangent{Transpose}(; parent=ZeroTangent()))
    end

    @testset "to Adjoint" begin
        x = rand(3, 4)
        a = adjoint(x)
        ma = collect(a)

        @test a == ProjectTo(a)(ma)
        @test zeros(4, 3) == ProjectTo(a)(ZeroTangent())
        @test zeros(4, 3) == ProjectTo(a)(Tangent{Adjoint}(; parent=ZeroTangent()))
    end

    @testset "to SubArray" begin
        x = rand(3, 4)
        sa = view(x, :, 1:2)
        m = collect(sa)

        @test m == ProjectTo(sa)(m)
        @test zeros(3, 2) == ProjectTo(sa)(ZeroTangent())
        @test_broken zeros(3, 2) == ProjectTo(sa)(Tangent{SubArray}(; parent=ZeroTangent())) # what do we want to do with SubArray?
    end
end
