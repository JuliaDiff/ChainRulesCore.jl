struct Foo
    a::Float64
end

Base.zero(::Foo) = Foo(0.0)
Base.zero(::Type{Foo}) = "F0"

@testset "projection" begin

    #identity
    @test Foo(1.2) == project(Foo(-0.2), Foo(1.2))
    @test 3.2 == project(1.0, 3.2)
    @test 2.0 + 0.0im == project(1.0im, 2.0)

    @testset "From AbstractZero" begin
        @testset "to numbers" begin
            @test 0.0 == project(1.1, ZeroTangent())
            @test 0.0f0 == project(1.1f0, ZeroTangent())
        end

        @testset "to arrays (dense and structured)" begin
            @test zeros(2, 2) == project([1.0 2; 3 4], ZeroTangent())
            @test Diagonal(zeros(2)) == project(Diagonal([1.0, 4]), ZeroTangent())
            @test Diagonal(zeros(ComplexF64, 2)) == project(Diagonal([1.0 + 0im, 4]), ZeroTangent())
        end

        @testset "to structs" begin
            @test Foo(0.0) == project(Foo(3.2), ZeroTangent())
        end

        @testset "to arrays of structs" begin
            @test [Foo(0.0), Foo(0.0)] == project([Foo(0.0), Foo(0.0)], ZeroTangent())
            @test Diagonal([Foo(0.0), Foo(0.0)]) == project(Diagonal([Foo(3.2,), Foo(4.2)]), ZeroTangent())
        end
    end

    @testset "From AbstractThunk" begin
        @test 3.2 == project(1.0, @thunk(3.2))
        @test Foo(3.2) == project(Foo(-0.2), @thunk(Foo(3.2)))
        @test zeros(2) == project([1.0, 2.0], @thunk(ZeroTangent()))
        @test Diagonal([Foo(0.0), Foo(0.0)]) == project(Diagonal([Foo(3.2,), Foo(4.2)]), @thunk(ZeroTangent()))
    end

    @testset "To number types" begin
        @testset "to subset" begin
            @test 3.2 == project(1.0, 3.2 + 3im)
            @test 3.2f0 == project(1.0f0, 3.2)
            @test 3.2f0 == project(1.0f0, 3.2 - 3im)
        end

        @testset "to superset" begin
            @test 2.0 + 0.0im == project(2.0 + 1.0im, 2.0)
            @test 2.0 == project(2.0, 2.0f0)
        end
    end

    @testset "To Arrays" begin
        # change eltype
        @test [1.0 2.0; 3.0 4.0] == project(zeros(2, 2), [1.0 2.0; 3.0 4.0])
        @test [1.0f0 2; 3 4] == project(zeros(Float32, 2, 2), [1.0 2; 3 4])

        # from a structured array
        @test [1.0 0; 0 4] == project(zeros(2, 2), Diagonal([1.0, 4]))

        # from an array of specials
        @test [Foo(0.0), Foo(0.0)] == project([Foo(0.0), Foo(0.0)], [ZeroTangent(), ZeroTangent()])
    end

    @testset "Diagonal" begin
        d = Diagonal([1.0, 4.0])
        t = Tangent{Diagonal}(;diag=[1.0, 4.0])
        @test d == project(d, [1.0 2; 3 4])
        @test d == project(d, t)
        @test project(Tangent, d, d) isa Tangent

        @test Diagonal([Foo(0.0), Foo(0.0)]) == project(Diagonal([Foo(3.2,), Foo(4.2)]), Diagonal([ZeroTangent(), ZeroTangent()]))
        @test Diagonal([Foo(0.0), Foo(0.0)]) == project(Diagonal([Foo(3.2,), Foo(4.2)]), @thunk(ZeroTangent()))
    end

    # how to project to Upper/Lower Symmetric
end
