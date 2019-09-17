#######
# Demo setup

cool(x) = x + 1
cool(x, y) = x + y + 1

# a rule we define so we can test rules
dummy_identity(x) = x
@scalar_rule(dummy_identity(x), One())

#######

_second(t) = Base.tuple_type_head(Base.tuple_type_tail(t))

@testset "frule and rrule" begin
    @test frule(cool, 1) === nothing
    @test frule(cool, 1; iscool=true) === nothing
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

    frx, cool_pushforward = frule(cool, 1)
    @test frx == 2
    @test cool_pushforward(NamedTuple(), 1) == 1
    rrx, cool_pullback = rrule(cool, 1)
    self, rr1 = cool_pullback(1)
    @test self == NO_FIELDS
    @test rrx == 2
    @test rr1 == 1
end


@testset "Basic Wirtinger scalar_rule" begin
    myabs2(x) = abs2(x)
    @scalar_rule(myabs2(x), Wirtinger(x', x))

    @testset "real input" begin
        # even though our rule was define in terms of Wirtinger,
        # pushforward result will be real as real (even if seed is Compex)

        x = rand(Float64)
        f, myabs2_pushforward = frule(myabs2, x)
        @test f === x^2

        Δ = One()
        df = @inferred myabs2_pushforward(NamedTuple(), Δ)
        @test df === x + x

        Δ = rand(Complex{Int64})
        df = @inferred myabs2_pushforward(NamedTuple(), Δ)
        @test df === Δ * (x + x)
    end

    @testset "complex input" begin
        z = rand(Complex{Float64})
        f, myabs2_pushforward = frule(myabs2, z)
        @test f === abs2(z)

        df = @inferred myabs2_pushforward(NamedTuple(), One())
        @test df === Wirtinger(z', z)

        Δ = rand(Complex{Int64})
        df = @inferred myabs2_pushforward(NamedTuple(), Δ)
        @test df === Wirtinger(Δ * z', Δ * z)
    end
end


@testset "Advanced Wirtinger @scalar_rule: abs_to_pow" begin
    # This is based on SimeonSchaub excellent example:
    # https://gist.github.com/simeonschaub/a6dfcd71336d863b3777093b3b8d9c97

    # This is much more complex than the previous case
    # as it has many different types
    # depending on input, and the output types do not always agree

    abs_to_pow(x, p) = abs(x)^p
    @scalar_rule(
        abs_to_pow(x::Real, p),
        (
            p == 0 ? Zero() : p * abs_to_pow(x, p-1) * sign(x),
            Ω * log(abs(x))
        )
    )

    @scalar_rule(
        abs_to_pow(x::Complex, p),
        @setup(u = abs(x)),
        (
            p == 0 ? Zero() : p * u^(p-1) * Wirtinger(x' / 2u, x / 2u),
            Ω * log(abs(x))
        )
    )


    f = abs_to_pow
    @testset "f($x, $p)" for (x, p) in Iterators.product(
        (2, 3.4, -2.1, -10+0im, 2.3-2im),
        (0, 1, 2, 4.3, -2.1, 1+.2im)
    )
        expected_type_df_dx =
            if iszero(p)
                Zero
            elseif typeof(x) <: Complex
                Wirtinger
            elseif typeof(p) <: Complex
                Complex
            else
                Real
            end

        expected_type_df_dp =
            if typeof(p) <: Real
                Real
            else
                Complex
            end


        res = frule(f, x, p)
        @test res !== nothing  # Check the rule was defined
        fx, f_pushforward = res
        df(Δx, Δp) = f_pushforward(NamedTuple(), Δx, Δp)

        df_dx::Thunk = df(One(), Zero())
        df_dp::Thunk = df(Zero(), One())
        @test fx == f(x, p)  # Check we still get the normal value, right
        @test df_dx() isa expected_type_df_dx
        @test df_dp() isa expected_type_df_dp


        res = rrule(f, x, p)
        @test res !== nothing  # Check the rule was defined
        fx, f_pullback = res
        dself, df_dx, df_dp = f_pullback(One())
        @test fx == f(x, p)  # Check we still get the normal value, right
        @test dself == NO_FIELDS
        @test df_dx() isa expected_type_df_dx
        @test df_dp() isa expected_type_df_dp
    end
end
