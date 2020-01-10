@testset "DoesNotExist" begin
    dne = DoesNotExist()
    @test_throws Exception extern(dne)
    @test dne + dne == dne
    @test dne + 1 == 1
    @test 1 + dne == 1
    @test dne * dne == dne
    @test dne * 1 == dne
    @test 1 * dne == dne

    @test Zero() + dne == dne
    @test dne + Zero() == dne

    @test Zero() * dne == Zero()
    @test dne * Zero() == Zero()

    for x in dne
        @test x === dne
    end
    @test broadcastable(dne) isa Ref{DoesNotExist}
    @test conj(dne) == dne
end
