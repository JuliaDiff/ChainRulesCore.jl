@testset "Differential Common" begin
    @testset "Refine Differential" begin
        @test refine_differential(typeof(1.0 + 1im), Wirtinger(2,2)) == Wirtinger(2,2)
        @test refine_differential(typeof([1.0 + 1im]), Wirtinger(2,2)) == Wirtinger(2,2)

        @test refine_differential(typeof(1.2), Wirtinger(2,2)) == 4
        @test refine_differential(typeof([1.2]), Wirtinger(2,2)) == 4

        # For most differentials, in most domains, this does nothing
        for der in (DoesNotExist(), @thunk(23), @thunk(Wirtinger(2,2)), [1 2], One(), Zero(), 0.0)
            for 𝒟 in typeof.((1.0 + 1im, [1.0 + 1im], 1.2, [1.2]))
                @test refine_differential(𝒟, der) === der
            end
        end
    end
end
