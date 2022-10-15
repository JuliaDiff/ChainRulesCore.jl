@testset "NotImplemented" begin
    @testset "NotImplemented" begin
        ni = ChainRulesCore.NotImplemented(
            @__MODULE__, LineNumberNode(@__LINE__, @__FILE__), "error"
        )
        ni2 = ChainRulesCore.NotImplemented(
            @__MODULE__, LineNumberNode(@__LINE__, @__FILE__), "error2"
        )
        x = rand()
        thunk = @thunk(x^2)

        # conjugate
        @test conj(ni) === ni

        # addition
        for a in (true, x, NoTangent(), ZeroTangent(), thunk)
            @test ni + a === ni
            @test a + ni === ni
        end
        @test +ni === ni
        @test ni + ni2 === ni
        @test ni2 + ni === ni2

        # multiplication, division, and dot product
        @test -ni == ni
        for a in (true, x, thunk)
            @test ni * a === ni
            @test a * ni === ni
            @test dot(ni, a) === ni
            @test dot(a, ni) === ni
        end
        for a in (NoTangent(), ZeroTangent())
            @test ni * a === a
            @test a * ni === a
            @test a / ni === a
            @test dot(ni, a) === a
            @test dot(a, ni) === a
        end
        @test ni * ni2 === ni
        @test ni2 * ni === ni2
        @test dot(ni, ni2) === ni
        @test dot(ni2, ni) === ni2

        # broadcasting
        @test ni .* x === ni
        @test x .* ni === ni
        @test broadcastable(ni) isa Ref{typeof(ni)}

        # unsupported operations
        E = ChainRulesCore.NotImplementedException
        for a in (true, x, NoTangent(), ZeroTangent(), thunk)
            @test_throws E ni - a
            @test_throws E a - ni
        end
        @test_throws E ni - ni2
        for a in (true, x, thunk)
            @test_throws E ni / a
            @test_throws E a / ni
        end
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
