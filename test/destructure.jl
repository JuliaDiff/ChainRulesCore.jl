using ChainRulesCore:
    destructure,
    Restructure,
    pushforward_of_destructure,
    pullback_of_destructure,
    pullback_of_restructure

function check_destructure(X::AbstractArray, dX)

    # Check that the round-trip in tangent / cotangent space is the identity function.
    pf_des = pushforward_of_destructure(X)
    pb_des = pullback_of_destructure(X)

    # dX_dense = pf_des(dX)
    # @test dX_dense ≈ pf_des(pb_des(dX_dense))

    # Check that the round-trip is the identity function.
    @test X ≈ Restructure(X)(destructure(X))

    # Check the rrule for destructure.
    pb_des = pullback_of_destructure(X)
    pb_res = pullback_of_restructure(X)
    dX_des = pb_res(dX)

    # I don't have access to test_approx here.
    # Since I need to test these, maybe chunks of this PR belong in ChainRules.jl.
    @test dX_des ≈ pb_res(pb_des(dX_des))

    # # Check the rrule for restructure.

end

@testset "destructure" begin
    check_destructure(randn(3, 3), randn(3, 3))
    check_destructure(Diagonal(randn(3)), Tangent{Diagonal}(diag=randn(3)))
    check_destructure(Symmetric(randn(3, 3)), Tangent{Symmetric}(data=randn(3, 3)))
end
