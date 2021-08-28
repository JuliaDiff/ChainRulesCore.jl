using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using LinearAlgebra
using Zygote

import ChainRulesCore: rrule, pullback_of_destructure, pullback_of_restructure

using ChainRulesCore: RuleConfig, wrap_natural_pullback

# All of the examples here involve new functions (`my_mul` etc) so that it's possible to
# ensure that Zygote's / ChainRules' existing adjoints don't get in the way.

# Example 1: matrix-matrix multiplication.

my_mul(A::AbstractMatrix, B::AbstractMatrix) = A * B

function rrule(config::RuleConfig, ::typeof(my_mul), A::AbstractMatrix, B::AbstractMatrix)
    C = A * B
    natural_pullback_for_mul(C̄) = NoTangent(), C̄ * B', A' * C̄
    return C, wrap_natural_pullback(config, natural_pullback_for_mul, C, A, B)
end

A = randn(4, 3);
B = Symmetric(randn(3, 3));
C, pb = Zygote.pullback(my_mul, A, B);

@assert C ≈ my_mul(A, B)

dC = randn(4, 3);
dA, dB_zg = pb(dC);
dB = Tangent{typeof(B)}(data=dB_zg.data);

# Test correctness.
dA_fd, dB_fd_sym = FiniteDifferences.j′vp(central_fdm(5, 1), my_mul, dC, A, B);

# to_vec doesn't know how to make `Tangent`s, so instead I map it to a `Tangent` manually.
dB_fd = Tangent{typeof(B)}(data=dB_fd_sym.data);

test_approx(dA, dA_fd)
test_approx(dB, dB_fd)



# pullbacks for `Real`s so that they play nicely with the utility functionality.

ChainRulesCore.pullback_of_destructure(config::RuleConfig, x::Real) = identity

ChainRulesCore.pullback_of_restructure(config::RuleConfig, x::Real) = identity


# Example 2: something where the output isn't a matrix.

my_sum(x::AbstractArray) = sum(x)

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(my_sum), x::AbstractArray)
    y = my_sum(x)
    natural_pullback_my_sum(ȳ::Real) = NoTangent(), fill(ȳ, size(x))
    return y, wrap_natural_pullback(config, natural_pullback_my_sum, y, x)
end

A = Symmetric(randn(2, 2))
y, pb = Zygote.pullback(my_sum, A)

test_approx(y, my_sum(A))

dy = randn()
dA_zg, = pb(dy)
dA = Tangent{typeof(A)}(data=dA_zg.data)

dA_fd_sym, = FiniteDifferences.j′vp(central_fdm(5, 1), my_sum, dy, A)
dA_fd = Tangent{typeof(A)}(data=dA_fd_sym.data)

test_approx(dA, dA_fd)





# Example 3: structured-input-structured-output

my_scale(a::Real, x::AbstractMatrix) = a * x

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(my_scale), a::Real, x::AbstractMatrix,
)
    y = my_scale(a, x)
    natural_pullback_my_scale(ȳ::AbstractMatrix) = NoTangent(), dot(ȳ, x), a * ȳ
    return y, wrap_natural_pullback(config, natural_pullback_my_scale, y, a, x)
end

# DENSE TEST
a = randn()
x = randn(2, 2)
y, pb = Zygote.pullback(my_scale, a, x)

dy = randn(size(y))
da, dx = pb(dy)

da_fd, dx_fd = FiniteDifferences.j′vp(central_fdm(5, 1), my_scale, dy, a, x)

test_approx(y, my_scale(a, x))
test_approx(da, da_fd)
test_approx(dx, dx_fd)



# DIAGONAL TEST

# `diag` now returns a `Diagonal` as a tangnet, so have to define `my_diag` to make this
# work with Diagonal`s.
my_diag(x) = diag(x)
function ChainRulesCore.rrule(::typeof(my_diag), D::P) where {P<:Diagonal}
    my_diag_pullback(d) = NoTangent(), Tangent{P}(diag=d)
    return diag(D), my_diag_pullback
end

a = randn()
x = Diagonal(randn(2))
y, pb = Zygote.pullback(my_diag ∘ my_scale, a, x)

ȳ = randn(2)
ā, x̄_zg = pb(ȳ)
x̄ = Tangent{typeof(x)}(diag=x̄_zg.diag)

ā_fd, _x̄_fd = FiniteDifferences.j′vp(central_fdm(5, 1), my_diag ∘ my_scale, ȳ, a, x)
x̄_fd = Tangent{typeof(x)}(diag=_x̄_fd.diag)

test_approx(y, (my_diag ∘ my_scale)(a, x))
test_approx(ā, ā_fd)
test_approx(x̄, x̄_fd)



# SYMMETRIC TEST - FAILS BECAUSE PRIVATE ELEMENTS IN LOWER-DIAGONAL ACCESSED IN PRIMAL!
# I would be surprised if we're doing this consistently at the minute though.

a = randn()
x = Symmetric(randn(2, 2))
y, pb = Zygote.pullback(my_scale, a, x)

dy = Tangent{typeof(y)}(data=randn(2, 2))
da, dx_zg = pb(dy)
dx = Tangent{typeof(x)}(data=dx_zg.data)

da_fd, dx_fd_sym = FiniteDifferences.j′vp(central_fdm(5, 1), my_scale, dy, a, x)
dx_fd = Tangent{typeof(x)}(data=dx_fd_sym.data)

test_approx(y.data, my_scale(a, x).data)
test_approx(da, da_fd)
test_approx(dx, dx_fd)





# Example 4: ScaledVector. This is an interesting example because I truly had no idea how to
# specify a natural tangent for this before.

# Implement AbstractArray interface.
struct ScaledMatrix <: AbstractMatrix{Float64}
    v::Matrix{Float64}
    α::Float64
end

Base.getindex(x::ScaledMatrix, p::Int, q::Int) = x.α * x.v[p, q]

Base.size(x::ScaledMatrix) = size(x.v)


# Implement destructure and restructure pullbacks.

function pullback_of_destructure(config::RuleConfig, x::P) where {P<:ScaledMatrix}
    function pullback_destructure_ScaledMatrix(X̄::AbstractArray)
        return Tangent{P}(v = X̄ * x.α, α = dot(X̄, x.v))
    end
    return pullback_destructure_ScaledMatrix
end

function pullback_of_restructure(config::RuleConfig, x::ScaledMatrix)
    function pullback_restructure_ScaledMatrix(x̄::Tangent)
        return x̄.v / x.α
    end
    return pullback_restructure_ScaledMatrix
end

# What destructure and restructure would look like if implemented. pullbacks were derived
# based on these.
# ChainRulesCore.destructure(x::ScaledMatrix) = x.α * x.v

# ChainRulesCore.Restructure(x::P) where {P<:ScaledMatrix} = Restructure{P, Float64}(x.α)

# (r::Restructure{<:ScaledMatrix})(x::AbstractArray) = ScaledMatrix(x ./ r.data, r.data)




# Define a function on the type.

my_dot(x::AbstractArray, y::AbstractArray) = dot(x, y)

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(my_dot), x::AbstractArray, y::AbstractArray,
)
    z = my_dot(x, y)
    natural_pullback_my_dot(z̄::Real) = NoTangent(), z̄ * y, z̄ * x
    return z, wrap_natural_pullback(config, natural_pullback_my_dot, z, x, y)
end


# Check correctness of `my_dot` rrule. Build `ScaledMatrix` internally to avoid technical
# issues with FiniteDifferences.
V = randn(2, 2)
α = randn()
z̄ = randn()

foo_scal(V, α) = my_dot(ScaledMatrix(V, α), V)

z, pb = Zygote.pullback(foo_scal, V, α)
dx_ad = pb(z̄)

dx_fd = FiniteDifferences.j′vp(central_fdm(5, 1), foo_scal, z̄, V, α)

test_approx(dx_ad, dx_fd)


# A function with a specialised method for ScaledMatrix.
my_scale(a::Real, X::ScaledMatrix) = ScaledMatrix(X.v, X.α * a)

# Verify correctness.
a = randn()
V = randn(2, 2)
α = randn()
z̄ = randn()

# A more complicated programme involving `my_scale`.
B = randn(2, 2)
foo_my_scale(a, V, α) = my_dot(B, my_scale(a, ScaledMatrix(V, α)))

z, pb = Zygote.pullback(foo_my_scale, a, V, α)
da, dV, dα = pb(z̄)

da_fd, dV_fd, dα_fd = FiniteDifferences.j′vp(central_fdm(5, 1), foo_my_scale, z̄, a, V, α)

test_approx(da, da_fd)
test_approx(dV, dV_fd)
test_approx(dα, dα_fd)
