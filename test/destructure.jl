using ChainRulesCore:
    destructure,
    Restructure,
    pushforward_of_destructure,
    pullback_of_destructure,
    pullback_of_restructure

# Need structural versions for tests, rather than the thing we currently have in
# FiniteDifferences.
function FiniteDifferences.to_vec(X::Diagonal)
    diag_vec, diag_from_vec = to_vec(X.diag)
    Diagonal_from_vec(diag_vec) = Diagonal(diag_from_vec(diag_vec))
    return diag_vec, Diagonal_from_vec
end

function FiniteDifferences.to_vec(X::Symmetric)
    data_vec, data_from_vec = to_vec(X.data)
    Symmetric_from_vec(data_vec) = Symmetric(data_from_vec(data_vec))
    return data_vec, Symmetric_from_vec
end

interpret_as_Tangent(x::Array) = x

interpret_as_Tangent(d::Diagonal) = Tangent{Diagonal}(diag=d.diag)

interpret_as_Tangent(s::Symmetric) = Tangent{Symmetric}(data=s.data)

Base.isapprox(t::Tangent{<:Diagonal}, d::Diagonal) = isapprox(t.diag, d.diag)

Base.isapprox(t::Tangent{<:Symmetric}, s::Symmetric) = isapprox(t.data, s.data)

function check_destructure(x::AbstractArray, ȳ, ẋ)

    # Verify correctness of frule.
    yf, ẏ = frule((NoTangent(), ẋ), destructure, x)
    @test yf ≈ destructure(x)

    ẏ_fd = jvp(central_fdm(5, 1), destructure, (x, ẋ))
    @test ẏ ≈ ẏ_fd

    yr, pb = rrule(destructure, x)
    _, x̄_r = pb(ȳ)
    @test yr ≈ destructure(x)

    # Use inner product relationship to avoid needing CRTU.
    @test dot(ȳ, ẏ_fd) ≈ dot(x̄_r, ẋ)

    # Check that the round-trip in tangent / cotangent space is the identity function.
    pf_des = pushforward_of_destructure(x)
    pb_des = pullback_of_destructure(x)

    # I thought that maybe the pushforward of destructure would be equivalent to the
    # pullback of restructure, but that doesn't seem to hold. Not sure why / whether I
    # should have thought it might be a thing in the first place.
    ẋ_dense = pf_des(ẋ)
    # @test ẋ_dense ≈ pf_des(pb_des(ẋ_dense))

    # Check that the round-trip is the identity function.
    @test x ≈ Restructure(x)(destructure(x))

    # Verify frule of restructure.
    x_dense = destructure(x)
    x_re, ẋ_re = frule((NoTangent(), ẋ_dense), Restructure(x), x_dense)
    @test x_re ≈ Restructure(x)(x_dense)

    ẋ_re_fd = FiniteDifferences.jvp(central_fdm(5, 1), Restructure(x), (x_dense, ẋ_dense))
    @test ẋ_re ≈ ẋ_re_fd

    # Verify rrule of restructure.
    x_re_r, pb_r = rrule(Restructure(x), x_dense)
    _, x̄_dense = pb_r(ẋ)

    # ẋ serves as the cotangent for the reconstructed x
    @test dot(ẋ, interpret_as_Tangent(ẋ_re_fd)) ≈ dot(x̄_dense, ẋ_dense)

    # Check the rrule for destructure.
    pb_des = pullback_of_destructure(x)
    pb_res = pullback_of_restructure(x)
    x̄_des = pb_res(ẋ)

    # I don't have access to test_approx here.
    # Since I need to test these, maybe chunks of this PR belong in ChainRules.jl.
    @test x̄_des ≈ pb_res(pb_des(x̄_des))
end

@testset "destructure" begin
    check_destructure(randn(3, 3), randn(3, 3), randn(3, 3))
    check_destructure(Diagonal(randn(3)), randn(3, 3), Tangent{Diagonal}(diag=randn(3)))
    check_destructure(
        Symmetric(randn(3, 3)), randn(3, 3), Tangent{Symmetric}(data=randn(3, 3)),
    )
end
