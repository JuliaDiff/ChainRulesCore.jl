using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using LinearAlgebra
using Zygote

import ChainRulesCore: rrule

using ChainRulesCore:
    pullback_of_destructure,
    pullback_of_restructure,
    RuleConfig,
    wrap_natural_pullback

# All of the examples here involve new functions (`my_mul` etc) so that it's possible to
# ensure that Zygote's existing adjoints don't get in the way.

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



# Example 2: something where the output isn't a matrix.
my_sum(x::AbstractArray) = sum(x)

function ChainRulesCore.rrule(::typeof(my_sum), x::Array)
    my_sum_strict_pullback(dy::Real) = (NoTangent(), dy * ones(size(x)))
    return sum(x), my_sum_strict_pullback
end

function ChainRulesCore.rrule(::typeof(my_sum), x::AbstractArray)
    x_dense, destructure_pb = ChainRulesCore.rrule(destructure, x)
    y, my_sum_strict_pullback = ChainRulesCore.rrule(my_sum, x_dense)

    function my_sum_generic_pullback(dy::Real)
        _, dx_dense = my_sum_strict_pullback(dy)
        _, dx = destructure_pb(dx_dense)
        return NoTangent(), dx
    end

    return y, my_sum_generic_pullback
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

function ChainRulesCore.rrule(::typeof(my_inv), x::Matrix)

    y, pb = ChainRulesCore.rrule(inv, x)

    # We know that a * x isa Array. Any AbstractArray is an okay tangent for an Array.
    function my_scale_pullback(ȳ::AbstractArray)
        return NoTangent(), dot(ȳ, x), ȳ * a
    end
    return a * x, my_scale_pullback
end

function ChainRulesCore.rrule(::typeof(my_scale), a::Real, x::AbstractMatrix)
    x_dense, destructure_x_pb = ChainRulesCore.rrule(destructure, x)
    y_dense, my_scale_strict_pb = ChainRulesCore.rrule(my_scale, a, x_dense)
    y = my_scale(a, x)
    y_reconstruct, restructure_pb = ChainRulesCore.rrule(Restructure(y), y_dense)

    function my_scale_generic_pullback(dy)
        _, dy_dense = restructure_pb(dy)
        _, da, dx_dense = my_scale_strict_pb(dy_dense)
        _, dx = destructure_x_pb(dx_dense)
        return NoTangent(), da, dx
    end

    return y_reconstruct, my_scale_generic_pullback
end

Zygote.refresh()

# SYMMETRIC TEST

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

# DENSE TEST
x_dense = collect(x)
y, pb = Zygote.pullback(my_scale, a, x_dense)

dy = randn(size(y))
da, dx = pb(dy)

da_fd, dx_fd = FiniteDifferences.j′vp(central_fdm(5, 1), my_scale, dy, a, x_dense)

test_approx(y, my_scale(a, x_dense))
test_approx(da, da_fd)
test_approx(dx, dx_fd)





# Example 4: ScaledVector

using ChainRulesCore
using ChainRulesCore: Restructure, destructure, Restructure
using ChainRulesTestUtils
using FiniteDifferences
using LinearAlgebra
using Zygote

# Implement AbstractArray interface.
struct ScaledMatrix <: AbstractMatrix{Float64}
    v::Matrix{Float64}
    α::Float64
end

Base.getindex(x::ScaledMatrix, p::Int, q::Int) = x.α * x.v[p, q]

Base.size(x::ScaledMatrix) = size(x.v)


# Implement destructure and restructure.

ChainRulesCore.destructure(x::ScaledMatrix) = x.α * x.v

ChainRulesCore.Restructure(x::P) where {P<:ScaledMatrix} = Restructure{P, Float64}(x.α)

(r::Restructure{<:ScaledMatrix})(x::AbstractArray) = ScaledMatrix(x ./ r.data, r.data)




# Define a function on the type.

my_dot(x::AbstractArray, y::AbstractArray) = dot(x, y)

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(my_dot), x::AbstractArray, y::AbstractArray,
)
    _, destructure_x_pb = rrule_via_ad(config, destructure, x)
    _, destructure_y_pb = rrule_via_ad(config, destructure, y)

    function pullback_my_dot(z̄::Real)
        x̄_dense = z̄ * y
        ȳ_dense = z̄ * x
        _, x̄ = destructure_x_pb(x̄_dense)
        _, ȳ = destructure_y_pb(ȳ_dense)
        return NoTangent(), x̄, ȳ
    end
    return my_dot(x, y), pullback_my_dot
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


# A function with a specialised rule for ScaledMatrix.
my_scale(a::Real, X::AbstractArray) = a * X
my_scale(a::Real, X::ScaledMatrix) = ScaledMatrix(X.v, X.α * a)

# Generic rrule.
function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(my_scale), a::Real, X::AbstractArray,
)
    _, destructure_X_pb = rrule_via_ad(config, destructure, X)
    Y = my_scale(a, X)
    _, restructure_Y_pb = rrule_via_ad(config, Restructure(Y), collect(Y))

    function pullback_my_scale(Ȳ)
        _, Ȳ_dense = restructure_Y_pb(Ȳ)
        ā = dot(Ȳ_dense, X)
        X̄_dense = Ȳ_dense * a
        _, X̄ = destructure_X_pb(X̄_dense)
        return NoTangent(), ā, X̄
    end

    return Y, pullback_my_scale
end

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





# Utility functionality.

# This will often make life really easy. Just requires that pullback_of_restructure is
# defined for C, and pullback_of_destructure for A and B. Could be generalised to make
# different assumptions (e.g. some arguments don't require destructuring, output doesn't
# require restructuring, etc). Would need to be generalised to arbitrary numbers of
# arguments (clearly doable -- at worst requires a generated function).
function wrap_natural_pullback(natural_pullback, C, A, B)

    # Generate enclosing pullbacks. Notice that C / A / B only appear here, and aren't
    # part of the closure returned. This means that they don't need to be carried around,
    # which is good.
    destructure_A_pb = pullback_of_destructure(A)
    destructure_B_pb = pullback_of_destructure(B)
    restructure_C_pb = pullback_of_restructure(C)

    # Wrap natural_pullback to make it play nicely with AD.
    function generic_pullback(C̄)
        _, C̄_natural = restructure_C_pb(C̄)
        f̄, Ā_natural, B̄_natural = natural_pullback(C̄_natural)
        _, Ā = destructure_A_pb(Ā_natural)
        _, B̄ = destructure_B_pb(B̄_natural)
        return f̄, Ā, B̄
    end
    return generic_pullback
end

# Sketch of rrule for my_mul making use of utility functionality.
function rrule(::typeof(my_mul), A::AbstractMatrix, B::AbstractMatrix)

    # Do the primal computation.
    C = A * B

    # "natural pullback"
    function my_mul_natural_pullback(C̄_natural)
        Ā_natural = C̄_natural * B'
        B̄_natural = A' * C̄_natural
        return NoTangent(), Ā_natural, B̄_natural
    end

    return C, wrap_natural_pullback(my_mul_natural_pullback, C, A, B)
end



# Order in which to present stuff.
# 1. Fully worked-through example (matrix-matrix) multiplication:
#   a. Most stupid implementation.
#   b. Optimal manual implementation.
#   c. Optimal implementation using utility functionality.
