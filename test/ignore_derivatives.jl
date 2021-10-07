struct MyFunctor
    a::Float64
end
(mf::MyFunctor)(b) = mf.a + b

@testset "ignore_derivatives.jl" begin
    @testset "function" begin
        f() = return 4.0

        y, ẏ = frule((1.0,), ignore_derivatives, f)
        @test y == f()
        @test ẏ == NoTangent()

        y, pb = rrule(ignore_derivatives, f)
        @test y == f()
        @test pb(1.0) == (NoTangent(), NoTangent())
    end

    @testset "argument" begin
        arg = 2.1

        y, ẏ = frule((1.0,), ignore_derivatives, arg)
        @test y == arg
        @test ẏ == NoTangent()

        y, pb = rrule(ignore_derivatives, arg)
        @test y == arg
        @test pb(1.0) == (NoTangent(), NoTangent())
    end

    @testset "functor" begin
        mf = MyFunctor(1.0)

        # as an argument
        y, ẏ = frule((1.0,), ignore_derivatives, mf)
        @test y == mf
        @test ẏ == NoTangent()

        y, pb = rrule(ignore_derivatives, mf)
        @test y == mf
        @test pb(1.0) == (NoTangent(), NoTangent())

        # when called
        y, ẏ = frule((1.0,), ignore_derivatives, () -> mf(3.0))
        @test y == mf(3.0)
        @test ẏ == NoTangent()

        y, pb = rrule(ignore_derivatives, () -> mf(3.0))
        @test y == mf(3.0)
        @test pb(1.0) == (NoTangent(), NoTangent())
    end
end
