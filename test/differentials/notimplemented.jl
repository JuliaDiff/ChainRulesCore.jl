@testset "NotImplemented" begin
    @testset "NotImplemented" begin
        ni = ChainRulesCore.NotImplemented(
            @__MODULE__, LineNumberNode(@__LINE__, @__FILE__), "error"
        )
        ni2 = ChainRulesCore.NotImplemented(
            @__MODULE__, LineNumberNode(@__LINE__, @__FILE__), "error2"
        )

        # supported operations (for `@scalar_rule`)
        x, y, z = rand(3)
        @test conj(ni) === ni
        @test muladd(ni, y, z) === ni
        @test muladd(ni, ZeroTangent(), z) == z
        @test muladd(ni, y, ZeroTangent()) === ni
        @test muladd(ni, ZeroTangent(), ZeroTangent()) == ZeroTangent()
        @test muladd(ni, ni2, z) === ni
        @test muladd(ni, ni2, ZeroTangent()) === ni
        @test muladd(ni, y, ni2) === ni
        @test muladd(ni, ZeroTangent(), ni2) === ni2
        @test muladd(x, ni, z) === ni
        @test muladd(ZeroTangent(), ni, z) == z
        @test muladd(x, ni, ZeroTangent()) === ni
        @test muladd(ZeroTangent(), ni, ZeroTangent()) == ZeroTangent()
        @test muladd(x, ni, ni2) === ni
        @test muladd(ZeroTangent(), ni, ni2) === ni2
        @test muladd(x, y, ni) === ni
        @test muladd(ZeroTangent(), y, ni) === ni
        @test muladd(x, ZeroTangent(), ni) === ni
        @test muladd(ZeroTangent(), ZeroTangent(), ni) === ni
        @test ni + rand() === ni
        @test ni + ZeroTangent() === ni
        @test ni + NoTangent() === ni
        @test ni + true === ni
        @test ni + @thunk(x^2) === ni
        @test rand() + ni === ni
        @test ZeroTangent() + ni === ni
        @test NoTangent() + ni === ni
        @test true + ni === ni
        @test @thunk(x^2) + ni === ni
        @test ni + ni2 === ni
        @test ni * rand() === ni
        @test ni * ZeroTangent() == ZeroTangent()
        @test ZeroTangent() * ni == ZeroTangent()
        @test dot(ni, ZeroTangent()) == ZeroTangent()
        @test dot(ZeroTangent(), ni) == ZeroTangent()
        @test ni .* rand() === ni
        @test broadcastable(ni) isa Ref{typeof(ni)}

        # unsupported operations
        E = ChainRulesCore.NotImplementedException
        @test_throws E +ni
        @test_throws E -ni
        @test_throws E ni - rand()
        @test_throws E ni - ZeroTangent()
        @test_throws E ni - NoTangent()
        @test_throws E ni - true
        @test_throws E ni - @thunk(x^2)
        @test_throws E rand() - ni
        @test_throws E ZeroTangent() - ni
        @test_throws E NoTangent() - ni
        @test_throws E true - ni
        @test_throws E @thunk(x^2) - ni
        @test_throws E ni - ni2
        @test_throws E rand() * ni
        @test_throws E NoTangent() * ni
        @test_throws E true * ni
        @test_throws E @thunk(x^2) * ni
        @test_throws E ni * ni2
        @test_throws E dot(ni, rand())
        @test_throws E dot(ni, NoTangent())
        @test_throws E dot(ni, true)
        @test_throws E dot(ni, @thunk(x^2))
        @test_throws E dot(rand(), ni)
        @test_throws E dot(NoTangent(), ni)
        @test_throws E dot(true, ni)
        @test_throws E dot(@thunk(x^2), ni)
        @test_throws E dot(ni, ni2)
        @test_throws E ni / rand()
        @test_throws E rand() / ni
        @test_throws E ni / ni2
        @test_throws E zero(ni)
        @test_throws E zero(typeof(ni))
        @test_throws E iterate(ni)
        @test_throws E iterate(ni, nothing)
        @test_throws E adjoint(ni)
        @test_throws E transpose(ni)
        @test_throws E convert(Float64, ni)
    end

    @testset "@not_implemented" begin
        ni = @not_implemented("myerror")
        @test ni isa ChainRulesCore.NotImplemented
        @test ni.mod isa Module
        @test ni.source isa LineNumberNode
        @test ni.info == "myerror"

        info = "some info"
        ni = @not_implemented(info)
        @test ni isa ChainRulesCore.NotImplemented
        @test ni.mod isa Module
        @test ni.source isa LineNumberNode
        @test ni.info === info
    end

    @testset "NotImplementedException" begin
        ni = @not_implemented("not implemented")
        ex = ChainRulesCore.NotImplementedException(ni)
        @test ex isa ChainRulesCore.NotImplementedException
        @test ex.mod === ni.mod
        @test ex.source === ni.source
        @test ex.info === ni.info
    end

    @testset "@scalar_rule" begin
        notimplemented1(x, y) = x + y
        @scalar_rule notimplemented1(x, y) (@not_implemented("notimplemented1"), 1)

        y, ẏ = frule((NoTangent(), 1.2, 2.3), notimplemented1, 3, 2)
        @test y == 5
        @test ẏ isa ChainRulesCore.NotImplemented

        res, pb = rrule(notimplemented1, 3, 2)
        @test res == 5
        f̄, x̄1, x̄2 = pb(3.1)
        @test f̄ == NoTangent()
        @test x̄1 isa ChainRulesCore.NotImplemented
        @test x̄2 == 3.1

        notimplemented2(x, y) = (x + y, x - y)
        @scalar_rule notimplemented2(x, y) (@not_implemented("notimplemented2"), 1) (1, -1)

        y, (ẏ1, ẏ2) = frule((NoTangent(), 1.2, 2.3), notimplemented2, 3, 2)
        @test y == (5, 1)
        @test ẏ1 isa ChainRulesCore.NotImplemented
        @test ẏ2 ≈ -1.1

        res, pb = rrule(notimplemented2, 3, 2)
        @test res == (5, 1)
        f̄, x̄1, x̄2 = pb((3.1, 4.5))
        @test f̄ == NoTangent()
        @test x̄1 isa ChainRulesCore.NotImplemented
        @test x̄2 == -1.4
    end
end
