using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using LinearAlgebra
using Zygote

function ChainRulesCore.rrule(::typeof(getindex), x::Symmetric, p::Int, q::Int)
    function structural_getindex_pullback(dy)
        ddata = zeros(size(x.data))
        if p > q
            ddata[q, p] = dy
        else
            ddata[p, q] = dy
        end
        return NoTangent(), Tangent{Symmetric}(data=ddata), NoTangent(), NoTangent()
    end
    return getindex(x, p, q), structural_getindex_pullback
end

function my_mul(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64})
    y1 = [Y[1, 1], Y[2, 1]]
    y2 = [Y[1, 2], Y[2, 2]]
    return reshape([X[1, :]'y1, X[2, :]'y1, X[1, :]'y2, X[2, :]'y2], 2, 2)
end

X = randn(2, 2)
Y = Symmetric(randn(2, 2))
Z, pb = Zygote.pullback(my_mul, X, Y)

Z̄ = randn(4)
X̄, Ȳ_zygote = pb(Z̄)

# Convert Ȳ to Tangent.
Ȳ = Tangent{typeof(Y)}(data=Ȳ_zygote.data)

# Essentially produces a structural tangent.
function FiniteDifferences.to_vec(X::Symmetric)
    x_vec, parent_from_vec = to_vec(X.data)
    function Symmetric_from_vec(x)
        return Symmetric(parent_from_vec(x))
    end
    return x_vec, Symmetric_from_vec
end

X̄_fd, Ȳ_fd_sym = FiniteDifferences.j′vp(central_fdm(5, 1), my_mul, Z̄, X, Y)

# to_vec doesn't know how to make `Tangent`s, so instead I map it to a `Tangent` manually.
Ȳ_fd = Tangent{typeof(Y)}(data=Ȳ_fd_sym.data)

Z_m, pb_m = Zygote.pullback(*, X, Y)
X̄_m, Ȳ_m = pb_m(reshape(Z̄, 2, 2))

# This is fine.
test_approx(X̄, X̄_fd)
@assert X + X̄ ≈ X + X̄_fd

# This is fine.
test_approx(X̄, X̄_m)
@assert X + X̄ ≈ X + X̄_m

# This is fine.
test_approx(Ȳ, Ȳ_fd)
@assert Y + Ȳ ≈ Y + Ȳ_fd

# This doesn't pass. To be expected, because Ȳ_m is a natural, and Ȳ a structural.
test_approx(Ȳ, Ȳ_m)
@assert Y + Ȳ_m ≈ Y + Ȳ_fd


A = randn(3, 2);
B = randn(2, 4);
C, pb = Zygote.pullback(*, A, B);
C̄ = randn(3, 4);
Ā, B̄ = pb(C̄);

Cm, pbm = Zygote.pullback(my_mul, A, B);
Ām, B̄m = pbm(C̄)

@assert C ≈ Cm
@assert Ā ≈ Ām
@assert B̄ ≈ B̄m

# Essentially the same as `collect`, but we get to choose semantics.
# I would imagine that `collect` will give us what we need most of the time, but sometimes
# it might not do what we want if e.g. the array in question lives on another device, or
# collect isn't implemented in a differentiable manner, or Zygote already implements the
# rrule for collect in a manner that confuses structurals and naturals.
destructure = collect

# I've had to implement a number of new functions here to ensure that they do things 
# structurally, because Zygote currently has a number of non-structural implementations
# of these things.

# THIS GUARANTEES ROUND-TRIP CONSISTENCY!

# IMPLEMENTATION 1: Very literal implementation. Not optimal, but hopefully the clearest 
# about what is going on.

using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using LinearAlgebra
using Zygote

# destructure is probably usually similar to collect, but we get to pick whatever semantics
# turn out to be useful.

# A real win of this approach is that we can test the correctness of people's
# destructure and restructure pullbacks using CRTU as per usual.
# We also just have a single simple requirement on the nature of destructure and
# restructure: restructure(destructure(X)) must be identical to X. Stronger than `==`.
function destructure(X::Symmetric)
    @assert X.uplo == 'U'
    return UpperTriangular(X.data) + UpperTriangular(X.data)' - Diagonal(X.data)
end

# Shouldn't need an rrule for this, since the above operations should all be fine, but
# Zygote currently has implementations of these that aren't structural, which is a problem.
function ChainRulesCore.rrule(::typeof(destructure), X::Symmetric)
    # As the type author in this context, I get to assert back type comes back.
    # I might also have chosen e.g. a GPUArray
    function destructure_pullback(dXm::Matrix)
        return NoTangent(), Tangent{Symmetric}(data=UpperTriangular(dXm) + LowerTriangular(dXm)' - Diagonal(dXm))
    end
    return destructure(X), destructure_pullback
end

destructure(X::Matrix) = X
function ChainRulesCore.rrule(::typeof(destructure), X::Matrix)
    destructure_pullback(dXm::Matrix) = NoTangent(), dXm
    return X, destructure_pullback
end

struct Restructure{P, D}
    required_primal_info::D
end

Restructure(X::P) where {P<:Matrix} = Restructure{P, Nothing}(nothing)

# Since the operation in question will return a `Matrix`, I don't need restructure for
# Symmetric matrices in this instance.
restructure(::Restructure{<:Matrix}, X::Matrix) = X

function ChainRulesCore.rrule(::typeof(restructure), ::Restructure{<:Matrix}, X::Matrix)
    restructure_matrix_pullback(dXm::Matrix) = NoTangent(), NoTangent(), dXm
    return X, restructure_matrix_pullback
end

Restructure(X::P) where {P<:Symmetric} = Restructure{P, Nothing}(nothing)

function restructure(r::Restructure{<:Symmetric}, X::Matrix)
    @assert issymmetric(X)
    return Symmetric(X)
end

function ChainRulesCore.rrule(::typeof(restructure), r::Restructure{<:Symmetric}, X::Matrix)
    function restructure_Symmetric_pullback(dX::Tangent)
        return NoTangent(), NoTangent(), dX.data
    end
    return restructure(r, X), restructure_Symmetric_pullback
end


my_mul(A::AbstractMatrix, B::AbstractMatrix) = A * B

function ChainRulesCore.rrule(::typeof(my_mul), A::Matrix, B::Matrix)
    function my_mul_pullback(C::Matrix)
        return NoTangent(), C * B', A' * C
    end
    return A * B, my_mul_pullback
end

# Could also use AD inside this definition.
function ChainRulesCore.rrule(::typeof(my_mul), A::AbstractMatrix, B::AbstractMatrix)

    # Produce dense versions of A and B, and the pullbacks of this operation.
    Am, destructure_A_pb = ChainRulesCore.rrule(destructure, A)
    Bm, destructure_B_pb = ChainRulesCore.rrule(destructure, B)

    # Compute the rrule in dense-land. This works by assumption.
    Cm, my_mul_strict_pullback = ChainRulesCore.rrule(my_mul, Am, Bm)

    # We need the output from the usual forwards pass in order to guarantee that we can
    # recover the correct structured type on the output side.
    C = my_mul(A, B)

    # Get the structured version back.
    _, restructure_C_pb = Zygote._pullback(Zygote.Context(), Restructure(C), Cm)

    # Note that I'm insisting on a `Tangent` here. Would also need to cover Thunks.
    function my_mul_generic_pullback(dC)
        _, dCm = restructure_C_pb(dC)
        _, dAm, dBm = my_mul_strict_pullback(dCm)
        _, dA = destructure_A_pb(dAm)
        _, dB = destructure_B_pb(dBm)
        return NoTangent(), dA, dB
    end

    return C, my_mul_generic_pullback
end

A = randn(4, 3)
B = Symmetric(randn(3, 3))
C, pb = Zygote.pullback(my_mul, A, B)

@assert C ≈ my_mul(A, B)

dC = randn(4, 3)
dA, dB_zg = pb(dC)
dB = Tangent{typeof(B)}(data=dB_zg.data)


# Test correctness.
dA_fd, dB_fd_sym = FiniteDifferences.j′vp(central_fdm(5, 1), my_mul, dC, A, B)

# to_vec doesn't know how to make `Tangent`s, so instead I map it to a `Tangent` manually.
dB_fd = Tangent{typeof(B)}(data=dB_fd_sym.data)

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

struct ScaledVector <: AbstractVector{Float64}
    v::Vector{Float64}
    α::Float64
end

Base.getindex(x::ScaledVector, n::Int) = x.α * x.v[n]

Base.size(x::ScaledVector) = size(x.v)

ChainRulesCore.destructure(x::ScaledVector) = x.α * x.v

ChainRulesCore.Restructure(x::P) where {P<:ScaledVector} = Restructure{P}(x.α)

(r::Restructure{<:ScaledVector})(x::AbstractVector) = ScaledVector(r.α, x ./ r.α)





# ALTERNATIVELY: just do forwards-mode through destructure, and avoid the need to implement
# restructure entirely. I think... hopefully this is correct?


# Under the current implementation, we have to do lots of things twice.
# Is there an implementation in which we don't have this problem?

function ChainRulesCore.rrule(::typeof(my_mul), A::AbstractMatrix, B::AbstractMatrix)

    # It's possible that our generic types have an optimised forwards pass involving
    # mutation. We would like to exploit this.
    C = A * B

    # Get the pullback for destructuring the arguments.
    # Currently actually does the destructuring, but this will often be entirely
    # unnecessary. Semantically, this is what we want though.
    _, destructure_A_pb = ChainRulesCore.rrule(destructure, A)
    _, destructure_B_pb = ChainRulesCore.rrule(destructure, B)

    # Somehow, we need to know how destructure would work, so that we can get its pullback.
    # This remains a very literal implementation -- it'll generally cheaper in practice.
    C_dense = destructure(C)
    _, restructure_C_pb = ChainRulesCore.rrule(restructure, Restructure(C), C_dense)

    # Restricted to Tangent for illustration purposes.
    # Does not permit a natural.
    # Thunk also needs to be supported.
    function my_mul_pullback(dC::Tangent)

        # Obtain natural tangent for output via restructure pullback.
        dC_natural = restructure_C_pb(dC)

        # Code to implement pullback in a generic manner, using natural tangents.
        dA_natural = dC_natural * B'
        dB_natural = A' * dC_natural

        # Obtain structural tangents for inputs via destructure pullback.
        _, dA = destructure_A_pb(dA_natural)
        _, dB = destructure_B_pb(dB_natural)

        return NoTangent(), dA, dB
    end

    return C, my_mul_pullback
end

# FORWARD-MODE AD THROUGH DESTRUCTURE YIELDS THE NATURAL TANGENT!
# The important thing is that e.g. a `Diagonal` is a valid tangent for a `Matrix`, because
# you can always find a matrix to which it is `==`.

# CLAIM: any AbstractArray is an acceptable tangent type for an Array.

# CLAIM: every AbstractArray has a natural tangent, induced by running forwards-mode on
# destructure.




# PR format:
1. basic claim -- find an equivalent programme, and work with that.
2.






In my on-going mission to figure out what these natural tangent things are really about, I've arrived at a scheme which gives us the following:

1. a generic construction for deriving generic rrules in terms of an equivalent primal programme,
2. a candidate method for formalising natural tangents as the result of doing AD on pieces of this equivalent primal programme.

The explanation of this PR will come in two chunks:
1. an explanation of the equivalent programme and its implications for natural tangents, and
2. approaches to optimising AD in the equivalent programme without changing its semantics.

I'm explaining it in this order because in my explanation of the programme, I'll have to run AD twice. This is useful for explaining what's going on, but isn't needed in practice.

# A Sketch of the Equivalent Programme

To begin with, consider a function
```julia
foo(x::AbstractArray)::AbstractArray
```
for which we want to write a generic `rrule`. To achieve this, assume that we have access to the following two functions:
```julia
destructure(::AbstractArray)::Array
```
defined such that
```julia
foo_equiv(x) = restructure(foo(x), foo(destructure(x)))
struct_isapprox(foo_equiv(x), foo(x))
```
for all `x`, where `struct_isapprox(a, b)` is defined to mean that all of the fields of `a` and `b` must be `struct_isapprox` with each other (i.e. the default version of `==` that people often ask for), with appropriate base-cases defined for non-composite types.

The first argument of `restructure` tells it what bit of data to aim for, and the second is the output of running `foo` on the `Array` which `==` `x`. Furthermore, assume that the pullback of `restructure` with respect to its first argument is always `ZeroTangent` or `NoTangent`, meaning that there's no need to AD back through it. This is the case for all of the types I've encountered so far, but I'm sure that there are types for which it will not be the case.

Hopefully it's clear that running AD on `foo_equiv` will yield the same answer as running AD on `foo`, up to known generic AD limitations (things like `x == 0 ? 0 : x` giving the wrong answer at `0`). Moreover, it is hopefully clear that `destructure` is trivial to implement -- `collect` will do. `restructure` is often simple -- for example,
```julia
restructure(D::Diagonal)
```



# Relaxing the Formulation a Bit


# Assumptions

1. Methods of `foo` specialised to specific subtypes of `AbstractArray` access via `getindex`. I believe we assume this implicitly generic rrules currently, and I believe that we need this assumption to guarantee correctness (I can construct an example that gives the wrong answer if this assumption is violated).
