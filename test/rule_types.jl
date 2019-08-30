
@testset "rule types" begin
    @testset "iterating and indexing rules" begin
        _, rule = frule(dummy_identity, 1)
        i = 0
        for r in rule
            @test r === rule
            i += 1
        end
        @test i == 1  # rules only iterate once, yielding themselves
        @test rule[1] == rule
        @test_throws BoundsError rule[2]
    end

    @testset "Rule" begin
        @testset "show" begin
            @test occursin(r"^Rule\(.*foo.*\)$", repr(Rule(function foo() 1 end)))
            @test occursin(r"^Rule\(.*identity.*\)$", repr(Rule(identity)))

            @test occursin(r"^Rule\(.*identity.*\,.*\+.*\)$", repr(Rule(identity, +)))
        end
    end

    @testset "WirtingerRule" begin
        myabs2(x) = abs2(x)

        function ChainRulesCore.frule(::typeof(myabs2), x)
            return abs2(x), AbstractRule(
                typeof(x),
                Rule(Δx -> Δx * x'),
                Rule(Δx -> Δx * x)
            )
        end

        # real input
        x = rand(Float64)
        f, _df = @inferred frule(myabs2, x)
        @test f === x^2

        df = @inferred _df(One())
        @test df === x + x


        Δ = rand(Complex{Int64})
        df = @inferred _df(Δ)
        @test df === Δ * (x + x)


        # complex input
        z = rand(Complex{Float64})
        f, _df = @inferred frule(myabs2, z)
        @test f === abs2(z)

        df = @inferred _df(One())
        @test df === Wirtinger(z', z)

        Δ = rand(Complex{Int64})
        df = @inferred _df(Δ)
        @test df === Wirtinger(Δ * z', Δ * z)
    end
end
