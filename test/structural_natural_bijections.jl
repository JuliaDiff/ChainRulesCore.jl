using ChainRulesCore: backing

# Need to be able to compare `Tangent`s to verify correctness of definition of natural.
Base.:(==)(t::Tangent, s::Tangent) = (backing(t) == backing(s))

# This function specifies what we require of a natural tangent. Works for flat spaces.
function check_bijections(primal, structural::Tangent)
    b = Bijections(primal)
    
    # Check that adding tangents is the same in structural or natural.
    sum_1 = 0.8 * structural + 0.3 * structural
    natural = to_natural(b, structural)
    sum_2 = to_structural(b, 0.8 * natural + 0.3 * natural)
    @test sum_1 == sum_2

    # Check that adding the structural tangent to the primal has the same effect as adding
    # the natural.
    @test primal + structural == primal + to_natural(b, structural)

    # Check that to_structural inverts to_natural.
    @test to_structural(b, to_natural(b, structural)) == structural
end

struct ScaledVector <: AbstractVector{Float64}
    v::Vector{Float64}
    α::Float64
end

Base.getindex(x::ScaledVector, n::Int) = x.α * x.v[n]

Base.size(x::ScaledVector) = size(x.v)

ChainRulesCore.Bijections(p::P) where {P<:ScaledVector} = Bijections{P, Float64}(p.α)

function ChainRulesCore.to_natural(b::Bijections{<:ScaledVector}, t::Tangent)
    return ScaledVector(t.v, t.α)
end

function ChainRulesCore.to_structural(
    b::Bijections{P}, n::ScaledVector,
) where {P<:ScaledVector}
    return Tangent{P}(v=n.v, α=n.α)
end

# Type-piracy.
LinearAlgebra.Symmetric(X::AbstractMatrix, uplo::Char) = Symmetric(X, Symbol(uplo))
LinearAlgebra.Hermitian(X::AbstractMatrix, uplo::Char) = Hermitian(X, Symbol(uplo))

# Bijections

@testset "structural_natural_bijections" begin
    x = randn(5)
    dx = randn(5)
    X = randn(5, 5)
    dX = randn(5, 5)

    check_bijections(Diagonal(x), Tangent{Diagonal}(diag=dx))
    check_bijections(UpperTriangular(X), Tangent{UpperTriangular}(data=dX))
    check_bijections(LowerTriangular(X), Tangent{LowerTriangular}(data=dX))
    check_bijections(Symmetric(X), Tangent{Symmetric}(data=dX))
    check_bijections(Hermitian(X), Tangent{Hermitian}(data=dX))

    # These tests don't work. I haven't managed to figure out a natural tangent which
    # satisfies all of my tests for this ScaledVector type.
    # check_bijections(ScaledVector(x, 0.5), Tangent{ScaledVector}(v=dx, α=randn()))
end
