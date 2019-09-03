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

    frx, fr = frule(cool, 1)
    @test frx == 2
    @test fr(NamedTuple(), 1) == (1,)
    rrx, (rr) = rrule(cool, 1)
    self, rr1 = rr(1)
    @test self == NO_FIELDS
    @test rrx == 2
    @test rr1 == 1
end


@testset "Wirtinger scalar_rule" begin
    myabs2(x) = abs2(x)
    @scalar_rule(myabs2(x), Wirtinger(x', x))

    # real input
    x = rand(Float64)
    f, pushforward = frule(myabs2, x)
    @test f === x^2

    df = @inferred pushforward(NamedTuple(), One())
    @test df === (x + x,)


    Δ = rand(Complex{Int64})
    df = @inferred pushforward(NamedTuple(), Δ)
    @test df === (Δ * (x + x),)


    # complex input
    z = rand(Complex{Float64})
    f, pushforward = frule(myabs2, z)
    @test f === abs2(z)

    df = @inferred pushforward(NamedTuple(), One())
    @test df === (Wirtinger(z', z),)

    Δ = rand(Complex{Int64})
    df = @inferred pushforward(NamedTuple(), Δ)
    @test df === (Wirtinger(Δ * z', Δ * z),)
end
