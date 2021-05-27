# Define some rules to test One on
dummy_identity(x) = x
@scalar_rule(dummy_identity(x), One())

very_nice(x, y) = x + y
@scalar_rule(very_nice(x, y), (One(), One()))

@testset "deprecations" begin
    @test ChainRulesCore.AbstractDifferential === ChainRulesCore.AbstractTangent
    @test Zero === ZeroTangent
    @test DoesNotExist === NoTangent
    @test Composite === Tangent
end

@testset "One()" begin

    o = One()
    @test extern(o) === true
    @test o + o == 2
    @test o + 1 == 2
    @test 1 + o == 2
    @test o * o == o
    @test o * 17 == 17
    @test 6 * o == 6
    @test dot(2 + im, o) == 2 - im
    @test dot(o, 2 + im) == 2 + im
    for x in o
        @test x === o
    end
    @test broadcastable(o) isa Ref{One}
    @test conj(o) == o

    @test reim(o) === (One(), ZeroTangent())
    @test real(o) === One()
    @test imag(o) === ZeroTangent()

    @test complex(o) === o
    @test complex(o, ZeroTangent()) === o
    @test complex(ZeroTangent(), o) === im

    @test frule((nothing, nothing, 5.0), Core._apply, dummy_identity, 4.0) == (4.0, 5.0)

    @testset "broadcasting One" begin
        sx = @SVector [1, 2]
        sy = @SVector [3, 4]

        # Test that @scalar_rule and `One()` play nice together, w.r.t broadcasting
        @inferred frule((ZeroTangent(), sx, sy), very_nice, 1, 2)
    end

    @testset "interaction with other types" begin
        c = Tangent{Foo}(y=1.5, x=2.5)
        @test One() * c === c
        @test c * One() === c

        z = ZeroTangent()
        @test zero(One()) === z
        @test zero(One) === z

        ni = ChainRulesCore.NotImplemented(
            @__MODULE__, LineNumberNode(@__LINE__, @__FILE__), "error"
        )
        @test ni + One() === ni
        @test One() + ni === ni
        E = ChainRulesCore.NotImplementedException
        @test_throws E ni - One()
        @test_throws E One() - ni
        @test_throws E One() * ni
        @test_throws E dot(ni, One())
        @test_throws E dot(One(), ni)
    end
end
