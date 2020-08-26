@testset "rule_definition_tools.jl" begin
    
    @testset "@non_differentiable" begin
        @testset "nondiff_2_1" begin
            nondiff_2_1(x, y) = fill(7.5, 100)[x + y]
            @non_differentiable nondiff_2_1(::Any, ::Any)
            @test frule((Zero(), 1.2, 2.3), nondiff_2_1, 3, 2) == (7.5, DoesNotExist())
            res, pullback = rrule(nondiff_2_1, 3, 2)
            @test res == 7.5
            @test pullback(4.5) == (NO_FIELDS, DoesNotExist(), DoesNotExist())
        end

        @testset "nondiff_1_2" begin
            nondiff_1_2(x) = (5.0, 3.0)
            @non_differentiable nondiff_1_2(::Any)
            @test frule((Zero(), 1.2), nondiff_1_2, 3.1) == ((5.0, 3.0), DoesNotExist())
            res, pullback = rrule(nondiff_1_2, 3.1)
            @test res == (5.0, 3.0)
            @test isequal(
                pullback(Composite{Tuple{Float64, Float64}}(1.2, 3.2)),
                (NO_FIELDS, DoesNotExist()),
            )
        end

        @testset "specific signature" begin
            nonembed_identity(x) = x
            @non_differentiable nonembed_identity(::Integer)

            @test frule((Zero(), 1.2), nonembed_identity, 2) == (2, DoesNotExist())
            @test frule((Zero(), 1.2), nonembed_identity, 2.0) == nothing

            res, pullback = rrule(nonembed_identity, 2)
            @test res == 2
            @test pullback(1.2) == (NO_FIELDS, DoesNotExist())

            @test rrule(nonembed_identity, 2.0) == nothing
        end
    end
end

