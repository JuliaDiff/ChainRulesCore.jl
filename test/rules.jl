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

#######

_second(t) = Base.tuple_type_head(Base.tuple_type_tail(t))

@testset "frule and rrule" begin
    dself = Zero()
    @test frule(cool, 1, dself, 1) === nothing
    @test frule(cool, 1, dself, 1; iscool=true) === nothing
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

    frx, cool_pushforward = frule(cool, 1, dself, 1)
    @test frx === 2
    @test cool_pushforward(dself, 1) === 1
    rrx, cool_pullback = rrule(cool, 1)
    self, rr1 = cool_pullback(1)
    @test self === NO_FIELDS
    @test rrx === 2
    @test rr1 === 1

    frx, nice_pushforward = frule(nice, 1, dself, 1)
    @test nice_pushforward(dself, 1) === Zero()
    rrx, nice_pullback = rrule(nice, 1)
    @test (dself, Zero()) === nice_pullback(1)

    sx = @SVector [1, 2]
    sy = @SVector [3, 4]
    _, fun = frule(very_nice, 1, 2, dself, sx, sy)
    # This actually is testing that @scalar_rule and `One()` play nice together, w.r.t broadcasting
    @inferred fun(dself, sx, sy)
end
