#######
# Demo setup

cool(x) = x + 1
cool(x, y) = x + y + 1

# a rule we define so we can test rules
dummy_identity(x) = x
@scalar_rule(dummy_identity(x), true)

nice(x) = 1
@scalar_rule(nice(x), ZeroTangent())

sum_two(x, y) = x + y
@scalar_rule(sum_two(x, y), (true, true))

complex_times(x) = (1 + 2im) * x
@scalar_rule(complex_times(x), 1 + 2im)

# Tests that aim to ensure that the API for frules doesn't regress and make these things
# hard to implement.

varargs_function(x...) = sum(x)
function ChainRulesCore.frule(dargs, ::typeof(varargs_function), x...)
    Δx = Base.tail(dargs)
    return sum(x), sum(Δx)
end

mixed_vararg(x, y, z...) = x + y + sum(z)
function ChainRulesCore.frule(
    dargs::Tuple{Any,Any,Any,Vararg}, ::typeof(mixed_vararg), x, y, z...
)
    Δx = dargs[2]
    Δy = dargs[3]
    Δz = dargs[4:end]
    return mixed_vararg(x, y, z...), Δx + Δy + sum(Δz)
end

type_constraints(x::Int, y::Float64) = x + y
function ChainRulesCore.frule(
    (_, Δx, Δy)::Tuple{Any,Int,Float64}, ::typeof(type_constraints), x::Int, y::Float64
)
    return type_constraints(x, y), Δx + Δy
end

mixed_vararg_type_constaint(x::Float64, y::Real, z::Vararg{Float64}) = x + y + sum(z)
function ChainRulesCore.frule(
    dargs::Tuple{Any,Float64,Real,Vararg{Float64}},
    ::typeof(mixed_vararg_type_constaint),
    x::Float64,
    y::Real,
    z::Vararg{Float64},
)
    Δx = dargs[2]
    Δy = dargs[3]
    Δz = dargs[4:end]
    return x + y + sum(z), Δx + Δy + sum(Δz)
end

ChainRulesCore.frule(dargs, ::typeof(Core._apply), f, x...) = frule(dargs[2:end], f, x...)

#######

_second(t) = Base.tuple_type_head(Base.tuple_type_tail(t))

@testset "frule and rrule" begin
    dself = ZeroTangent()
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
    only_methods = Set([
        Tuple{typeof(rrule),typeof(cool),Number}, Tuple{typeof(rrule),typeof(cool),String}
    ])
    @test cool_methods == only_methods

    frx, cool_pushforward = frule((dself, 1), cool, 1)
    @test frx === 2
    @test cool_pushforward === 1
    rrx, cool_pullback = rrule(cool, 1)
    self, rr1 = cool_pullback(1)
    @test self === NoTangent()
    @test rrx === 2
    @test rr1 == 1.0   # projection may make this ===

    frx, nice_pushforward = frule((dself, 1), nice, 1)
    @test nice_pushforward === ZeroTangent()
    rrx, nice_pullback = rrule(nice, 1)
    @test (NoTangent(), ZeroTangent()) === nice_pullback(1)

    # Test that these run. Do not care about numerical correctness.
    @test frule((nothing, 1.0, 1.0, 1.0), varargs_function, 0.5, 0.5, 0.5) == (1.5, 3.0)

    @test frule((nothing, 1.0, 2.0, 3.0, 4.0), mixed_vararg, 1.0, 2.0, 3.0, 4.0) ==
          (10.0, 10.0)

    @test frule((nothing, 3, 2.0), type_constraints, 5, 4.0) == (9.0, 5.0)
    @test frule((nothing, 3.0, 2.0im), type_constraints, 5, 4.0) == nothing

    @test(
        frule(
            (nothing, 3.0, 2.0, 1.0, 0.0), mixed_vararg_type_constaint, 3.0, 2.0, 1.0, 0.0
        ) == (6.0, 6.0)
    )

    # violates type constraints, thus an frule should not be found.
    @test frule((nothing, 3, 2.0, 1.0, 5.0), mixed_vararg_type_constaint, 3, 2.0, 1.0, 0) ==
          nothing

    @test frule((nothing, nothing, 5.0), Core._apply, dummy_identity, 4.0) == (4.0, 5.0)

    @testset "broadcasting true" begin
        sx = @SVector [1, 2]
        sy = @SVector [3, 4]

        # Test that @scalar_rule and `true` play nice together, w.r.t broadcasting
        @inferred frule((ZeroTangent(), sx, sy), sum_two, 1, 2)
    end

    @testset "complex numbers" begin
        x, ẋ, Ω̄ = randn(ComplexF64, 3)
        Ω = complex_times(x)

        # forwards
        Ω_fwd, Ω̇ = frule((nothing, ẋ), complex_times, x)
        @test Ω_fwd == Ω
        @test Ω̇ ≈ (1 + 2im) * ẋ

        # reverse
        Ω_rev, back = rrule(complex_times, x)
        @test Ω_rev == Ω
        ∂self, ∂x = back(Ω̄)
        @test ∂self == NoTangent()
        @test ∂x ≈ (1 - 2im) * Ω̄

        # real argument, complex output
        xr = rand()
        Ωr = complex_times(xr)
        Ωr_rev, backr = rrule(complex_times, xr)
        ∂selfr, ∂xr = backr(Ω̄)
        @test_skip ∂xr isa Float64  # to be made true with projection
        @test_skip ∂xr ≈ real(∂x)
    end

    @testset "@opt_out" begin
        first_oa(x, y) = x
        @scalar_rule(first_oa(x, y), (1, 0))
        @opt_out ChainRulesCore.rrule(::typeof(first_oa), x::T, y::T) where {T<:Float32}
        @opt_out(
            ChainRulesCore.frule(::Any, ::typeof(first_oa), x::T, y::T) where {T<:Float32}
        )

        @testset "rrule" begin
            @test rrule(first_oa, 3.0, 4.0)[2](1) == (NoTangent(), 1, 0)
            @test rrule(first_oa, 3.0f0, 4.0f0) === nothing

            @test !isempty(
                Iterators.filter(methods(ChainRulesCore.no_rrule)) do m
                    m.sig <: Tuple{Any,typeof(first_oa),T,T} where {T<:Float32}
                end,
            )
        end

        @testset "frule" begin
            @test frule((NoTangent(), 1, 0), first_oa, 3.0, 4.0) == (3.0, 1)
            @test frule((NoTangent(), 1, 0), first_oa, 3.0f0, 4.0f0) === nothing

            @test !isempty(
                Iterators.filter(methods(ChainRulesCore.no_frule)) do m
                    m.sig <: Tuple{Any,Any,typeof(first_oa),T,T} where {T<:Float32}
                end,
            )
        end
    end
end
