#######
# Demo setup
using StaticArrays: @SVector

cool(x) = x + 1
cool(x, y) = x + y + 1

# a rule we define so we can test rules
dummy_identity(x) = x
@scalar_rule(dummy_identity(x), One())

nice(x) = 1
@scalar_rule(nice(x), Zero())

very_nice(x, y) = x + y
@scalar_rule(very_nice(x, y), (One(), One()))

complex_times(x) = (1 + 2im) * x
@scalar_rule(complex_times(x), 1 + 2im)

# Tests that aim to ensure that the API for frules doesn't regress and make these things
# hard to implement.

varargs_function(x...) = sum(x)
@frule function ChainRulesCore.frule(dargs, ::typeof(varargs_function), x...)
    Δx = Base.tail(dargs)
    return sum(x), sum(Δx)
end

mixed_vararg(x, y, z...) = x + y + sum(z)
@frule function ChainRulesCore.frule(
    dargs::Tuple{Any, Any, Any, Vararg},
    ::typeof(mixed_vararg), x, y, z...,
)
    Δx = dargs[2]
    Δy = dargs[3]
    Δz = dargs[4:end]
    return mixed_vararg(x, y, z...), Δx + Δy + sum(Δz)
end

type_constraints(x::Int, y::Float64) = x + y
@frule function ChainRulesCore.frule(
    (_, Δx, Δy)::Tuple{Any, Int, Float64},
    ::typeof(type_constraints), x::Int, y::Float64,
)
    return type_constraints(x, y), Δx + Δy
end

mixed_vararg_type_constaint(x::Float64, y::Real, z::Vararg{Float64}) = x + y + sum(z)
@frule function ChainRulesCore.frule(
    dargs::Tuple{Any, Float64, Real, Vararg{Float64}},
    ::typeof(mixed_vararg_type_constaint), x::Float64, y::Real, z::Vararg{Float64},
)
    Δx = dargs[2]
    Δy = dargs[3]
    Δz = dargs[4:end]
    return x + y + sum(z), Δx + Δy + sum(Δz)
end

@frule function ChainRulesCore.frule(dargs, ::typeof(Core._apply), f, x...)
    return frule(dargs[2:end], f, x...)
end

#######

_second(t) = Base.tuple_type_head(Base.tuple_type_tail(t))

@testset "frule and rrule" begin
    dself = Zero()
    @test frule((dself, 1), cool, 1) === nothing
    @test frule((dself, 1), cool, 1; iscool=true) === nothing
    @test rrule(cool, 1) === nothing
    @test rrule(cool, 1; iscool=true) === nothing

    # add some methods:
    ChainRulesCore.@scalar_rule(Main.cool(x), one(x))
    @test hasmethod(rrule, Tuple{typeof(cool),Number})
    ChainRulesCore.@scalar_rule(Main.cool(x::String), "wow such dfdx")
    @test hasmethod(rrule, Tuple{typeof(cool),String})
    # Ensure those are the *only* methods that have been defined
    cool_methods = Set(m.sig for m in methods(rrule) if _second(m.sig) == typeof(cool))
    only_methods = Set([Tuple{typeof(rrule),typeof(cool),Number},
                        Tuple{typeof(rrule),typeof(cool),String}])
    @test cool_methods == only_methods

    frx, cool_pushforward = frule((dself, 1), cool, 1)
    @test frx === 2
    @test cool_pushforward === 1
    rrx, cool_pullback = rrule(cool, 1)
    self, rr1 = cool_pullback(1)
    @test self === NO_FIELDS
    @test rrx === 2
    @test rr1 === 1

    frx, nice_pushforward = frule((dself, 1), nice, 1)
    @test nice_pushforward === Zero()
    rrx, nice_pullback = rrule(nice, 1)
    @test (NO_FIELDS, Zero()) === nice_pullback(1)


    # Test that these run. Do not care about numerical correctness.
    @test frule((nothing, 1.0, 1.0, 1.0), varargs_function, 0.5, 0.5, 0.5) == (1.5, 3.0)

    @test frule((nothing, 1.0, 2.0, 3.0, 4.0), mixed_vararg, 1.0, 2.0, 3.0, 4.0) == (10.0, 10.0)

    @test frule((nothing, 3, 2.0), type_constraints, 5, 4.0) == (9.0, 5.0)
    @test frule((nothing, 3.0, 2.0im), type_constraints, 5, 4.0) == nothing

    @test(frule(
        (nothing, 3.0, 2.0, 1.0, 0.0),
        mixed_vararg_type_constaint, 3.0, 2.0, 1.0, 0.0,
    ) == (6.0, 6.0))

    # violates type constraints, thus an frule should not be found.
    @test frule(
        (nothing, 3, 2.0, 1.0, 5.0),
        mixed_vararg_type_constaint, 3, 2.0, 1.0, 0,
    ) == nothing

    @test frule((nothing, nothing, 5.0), Core._apply, dummy_identity, 4.0) == (4.0, 5.0)

    @testset "broadcasting One" begin
        sx = @SVector [1, 2]
        sy = @SVector [3, 4]

        # Test that @scalar_rule and `One()` play nice together, w.r.t broadcasting
        @inferred frule((Zero(), sx, sy), very_nice, 1, 2)
    end

    @testset "complex inputs" begin
        x, ẋ, Ω̄ = randn(ComplexF64, 3)
        Ω = complex_times(x)
        Ω_fwd, Ω̇ = frule((nothing, ẋ), complex_times, x)
        @test Ω_fwd == Ω
        @test Ω̇ ≈ jvp(central_fdm(5, 1), complex_times, (x, ẋ))
        Ω_rev, back = rrule(complex_times, x)
        @test Ω_rev == Ω
        ∂self, ∂x = back(Ω̄)
        @test ∂self == NO_FIELDS
        @test ∂x ≈ j′vp(central_fdm(5, 1), complex_times, Ω̄, x)[1]
    end
end


simo(x) = (x, 2x)
@scalar_rule(simo(x), 1, 2)

@testset "@scalar_rule with multiple inputs" begin
    y, simo_pb = rrule(simo, π)

    @test simo_pb((10, 20)) == (NO_FIELDS, 50)

    y, ẏ = frule((NO_FIELDS, 50), simo, π)
    @test y == (π, 2π)
    @test ẏ == (50, 100)
end
