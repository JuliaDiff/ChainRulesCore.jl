# The AbstractArray Dilemma


## The Aim Of This Document

We seek to provide the information necessary to answer the question: "is it acceptable /
desirable to treat `struct`s which subtype AbstractArray in the same way as other structs,
or must they be treated separately."
So after reading this document, you should feel better equipped to form an opinion on this
matter.


## An Example Problem

Consider an `rrule` with signature
```julia
function rrule(::typeof(*), A::AbstractMatrix, B::AbstractMatrix)
    C = A * B
    function mul_pullback(ΔC)
        ΔA = ΔC * B
        ΔB = A * ΔC'
        return NO_FIELDS, ΔA, ΔB
    end
    return C, mul_pullback
end
```
and consider its performance when `A` is a `Diagonal{Float64, Vector{Float64}}`,
and `B` a `Matrix{Float64}`.
`C` will be another `Matrix{Float64}`, meaning that the cotangent `ΔC` will usually be a
`Matrix{Float64}`.
This means that the cotangent `ΔA` will be a `Matrix{Float64}`!
Let `M := size(A, 1)` then the reverse-pass will require at least `M^2` `Float64`s, while
the primal pass will only require `M`.
Similarly, letting `N := size(B, 2)`, computing `ΔA` under this implementation requires
`O(M^2N)` operations, while the forwards requires only `O(MN)`.
Thus, if an AD system utilises this particular `rrule`, it will take (asymptotically) more
compute to run reverse-mode than to evaluate the function.
This will be particularly acute if `A` is large, and `B` is thin.


If, as is often the case, a user's reason for expressing a function in terms of a `Diagonal`
(rather than a `Matrix` which happens to be diagonal) is at least partly
performance-related, then this poor performance is crucially important.
To see this, observe that the time and memory required to execute the reverse-pass is
roughly the same as would be required to execute a primal pass had `A` been dense.
Thus, and performance gain that one might have hoped to obtain using a `Diagonal` has been
lost.

A possible way to rectify this problem is to require that only the diagonal of `ΔA` be
preserved, in which case the same asymptotic complexity can be obtained in the rrule as in
the function evaluation itself.
Indeed, if this rule did not exist, and the AD tool applied to `A * B` were successfully
able to derive a pullback, this is essentially what it will do.
Moreover, if `A` we some other `struct`, which was not to be interpretted as an array of
any kind, and it had the above `*` function defined on it, this is what AD would do.

You might reasonable wonder what other types this kind of thing occurs for.
It seems inevitable that all of the following will suffer from the same kinds of dilemma
for all functions involving them, each of which are commonly used to accelerate code:
1. `BiDiagonal`, `TriDiagonal`, `SymTridiagonal` (LinearAlgebra)
1. `SparseMatrixCSC` (SparseArrays)
1. `Fill`, `One`, `Zero` (FillArrays.jl)
1. `BlockDiagonal` (BlockDiagonals.jl)
1. `StructArray` (StructArrays.jl)
1. `WoodburyPDMat` (PDMatsExtras)
1. `PDiagMat`, `PDSparseMat`, `ScalMat` (PDMats.jl)
1. `InfiniteArray` (InfiniteArrays.jl)

If any of these types are placed inside the various wrapper types that Julia
offers (`UpperTriangular`, `LowerTriangular`, `Diagonal`, `Symmetric`, etc), the same kinds
of problems arise.
The same is true of things like `Matrix{<:Diagonal}` i.e. a matrix of `Diagonal` matrices.

In a very literal sense, the number of possible arrays that this could affect is unbounded,
because Julia users keep writing new array types which express particular types of
structure!
It's commonly the case that you think you've written code that your preferred AD tool should
be able to handle quite straightforwardly, only to find that a rule exists that gets in the
way of AD doing its job.
When this happens, it is more than a little frustrating!

However, we _do_ need rules for some "primitive" array types.
For example, the above rule works just fine for "dense" arrays such as `Matrix{Float64}`,
`StridedMatrix{Float64}`, `CuArray{Float32}`, `SMatrix{2, 2, Float64}`, as well as for other
real element types.
Simiarly, when someone defines a new array type for which the above implementation of `*`
and other similar functions works well / is a good choice, it would be unfortunate if they
don't have a quick way to declare their array as a "primitive" array, and to ensure that it
inherits these rules.

Hopefully, this section convinces you that there is something worth thinking about here, and
that this is not a problem which can be ignored.








## What Does Reverse-Mode AD Do?

Before considering what to do, it's helpful to have a working definition of reverse-mode AD.

Define a function `reverse_mode_ad` in terms of its relationship with a function
`forwards_mode_ad`:
```julia
ẏ = forwards_mode_ad(f, x, ẋ)
x̄ = reverse_mode_ad(f, x, ȳ)
<ȳ, ẏ> == <x̄, ẋ>

```
where `<., .>` is an inner product -- we'll usually write `dot` later on.
Therefore, if we have defined `forwards_mode_ad`, we have defined `reverse_mode_ad`.

We call the set of values that `ẋ` can take the _tangent_ _space_ _of_ `x`, and the set of
values that `x̄` can take the _cotangent_ _space_ _of_ `x`. The set of values that `ẏ` and
`ȳ` are called the _tangent_ and _cotangent_ spaces of `y` respectively.

For example, assuming `forwards_mode_ad` computes `ẏ = J(f, x) ẋ`, where `J` is the Jacobian
of `f` at `x`, then `x̄ = J(f, x)'ȳ`.
So in this situation, `reverse_mode_ad` can be used to compute the Jacobian.

However, if this assumption about `forwards_mode_ad` fails to hold for some `f` and
`x`, then neither `forwards_mode_ad` nor `reverse_mode_ad` will compute the above functions
of the Jacobian.
[This happens in practice](@ref some_peculiar_behaviour).







## Dense (Co)Tangent Spaces For Sparse Arrays

Equipped with an understanding of the relationship between forwards- and reverse-mode AD, we
turn to the primary question.

Let us first consider the consequences of letting the (co)tangent space of a sparse array be
equivalent to a dense for functions of a sparse array.
We will focus on `Diagonal` matrices, and `Matrix` (co)tangent spaces, but the arguments
below follow for any kind of array in which it can be crucial to exploit sparsity for
performance reasons, such as those listed above.

Under this definition of the (co)tangent space, reverse-mode AD must _always_ assume that
the tangent of a given
`Diagonal` matrix might have non-zero off-diagonals, and therefore must never drop the
off-diagonals in its cotangents.
For example, let `x` be a `Diagonal{<:Real}` matrix, `ẋ` a `Matrix{<:Real}` representing
a tangent for `x`, then for some function `f` and output-cotangent `ȳ`, then
```julia
dot(reverse_mode_ad(f, x, ȳ), ẋ)
```
involves the off-diagonal elements of `ẋ`, so the off-diagonal elements of
`reverse_mode_ad(f, x, ȳ)` must be retained if reverse-mode is to agree with forwards-mode.

Conversely, if the tangent space of `x` is restricted to be the `Diagonal` matrices, then
the off-diagonal elements of `ẋ` are always zero, and it's safe to drop the off-diagonal
elements of `reverse_mode_ad(f, x, ȳ)`.
A similar result if `Diagonal` is treated as a `struct`, and its tangent
type therefore required to be a `Composite`, since they're isomorphic to one another.







### [Functions of Sparse Arrays](@id functions_of_sparse_arrays)

To see how the above manifests itself in practice, consider the following function of a
`Diagonal` matrix:
```julia
h(D::AbstractMatrix{<:Real}, x::Vector{<:Real}) = (logdet(cholesky(D)), sum(D), D * x)
```
This is a perfectly plausible function -- it's the kind of thing you might see somewhere
buried inside code that deals with multivariate Normal distributions with covariance matrix
`D`.

If you evaluate it using a `Diagonal` matrix and a `Vector`
```julia
N = 1_000_000
h(Diagonal(rand(N) .+ 1), randn(N))
```
you obtain a function which requires O(N) operations, rather than the O(N^3 + N^2) required
if you evaluate it for a `Matrix` and a `Vector`.

If the semantics of AD are such that it is necessary to assume that the tangent of `D`
might be non-diagonal, it is necessary for AD to treat `D` like a `Matrix` which happens to
be diagonal, as discussed above.
This code will clearly not run if that is the case.

This is a very standard example of taking advantage of Julia's zero-overhead
abstractions to accelerate computation where possible without re-writing your code, and is
exactly what you would want if working with a multivariate Gaussian with diagonal covariance
matrix.
These kinds of structure occur regularly in practice -- perhaps you wish to compare Gaussian
models with a dense covariance matrix with different covariance structures (diagonal,
nearly-low-rank, sparse-inverse, etc) on the same data set to infer something interpretable
about the properties of your data.
In this case, linear algebra operations which exploit this structure is a nice-to-have, and
can accelerate your code a lot.

These accelerations become utterly essential when dealing with very
high-dimensional problems, such as the one above, in which you absolutely _need_ to restrict
your covariance matrix in some way to produce something tractable (e.g. there are entire
communities of people that work on this kind of thing in the context of Gaussian processes).

Furthermore, this choice of tangent space for `D` affords us no new
functionality as, if the implementer of `h` were interested in matrices which _happened_
to be diagonal (at the initialisation of some iterative algorithm, say) they could just
`collect` said matrix.







#### Can Abstraction Save Us?

As an aside, perhaps we could avoid treating the (co)tangent space of `D` as a `Matrix`, and
instead exploit some structure.
For example, the cotangent of `sum(D)` is might be represented by a `Fill`, `D * x` by a
rank-1 matrix, and perhaps the cotangent of `logdet(cholesky(D))` by... something?
One would then need to know how to add `AbstractMatrix`s of these types together in a manner
that is efficient and retains the structure, if a `Matrix` is to be avoided at the
accumulation step of reverse-mode AD.

Perhaps this is possible, but it seems incredibly _hard_, and is certainly not something
that could easily generalise to new structured `AbstractArray` types -- consider that an
author of a new `AbstractArray` would need to anticipate all of the operations that might be
defined on their matrix (that is clearly not possible), either utilise or define appropriate
`AbstractArray` types to represent the cotangent produced by each of them (which seems hard,
and might not be possible in general), and finally to know how to add these types together
(again, hard and certainly not easy to automate).

It is for these reasons that I do not believe this appoach to be plausible -- we must assume
that a `Matrix` is necessary (or some other "dense" array like a `CuArray`) if the
(co)tangent space is not restricted.







### [Intermediate Functions of Sparse Arrays](@id intermediate_functions)

In the previous example, the restrictions that appear to be necessary change the output of
AD from that which we might reasonable expect, which is probably a bit surprising for new
users.
However, it's very common (certainly not less common than the previous case) to find the
above example inside other code, as intermediate computations in some larger function.
In such cases, the restriction of the tangent space of a sparse array will _not_ change the
outcome, so in such situations we really lose nothing by restricting the tangent space, but
gain a lot.

For example, consider
```julia
function g(d::Vector{<:Real}, x::Vector{<:Real})
    D = Diagonal(d)
    return h(D, x)
end
```
The off-diagonal elements of the cotangent of `Diagonal(d)` are clearly not required in this
case due to the call to `Diagonal`, which necessarily ignores them on the reverse-pass.
However, `h` does not know this -- it does not get to know the context in which it is being
called -- therefore it _must_ assume that the cotangent needs to be non-diagonal if we allow
the for non-diagonal tangents of `D`.
For large `N`, this is clearly prohibitive, and precludes AD from being run on
`g`, despite `g` being a completely reasonable function to evaluate on a large `Diagonal`
matrix.







### Input-Dependent Intermediate Functions of Sparse Arrays 

The above example is quite clear-cut -- if the `Diagonal` is _always_ constructed somewhere
inside the function, then there's no need to keep the off diagonals.
This in turn led us to the conclusion that the off-diagonals _must_ be dropped to make such
programmes tractable to differentiate, and that the off-diagonals must therefore always be
dropped.

A slightly different situation occurs when the `Diagonal` is constructed in a manner which
depends in a particular way on the arguments of the function. We explore this situation
extensively elsewhere, in [Some Peculiar AD Behaviour](@ref some_peculiar_behaviour).

The upshot is that these situations are a property of AD generally, not specifically a
property of sparse linear algebra, so we consider them no further.







### Takeaways

Similar arguments could be constructed for the other structured linear algebra
types discussed earlier, so the lesson of this example seems to be that in order for
users of AD to be able to use structured linear algebra types in the usual way, it is
generally _necessary_ to place strong restrictions on the tangent space, and these
constraints _will_ lead to results that new users probably won't expect.

The consequence of not accepting these restrictions is that important programmes that really
ought to be straightforward to apply AD to are rendered intractable, for no notable increase
in the functionality available.

Moreover, the restrictions on the (co)tangent spaces of such types are what AD would
naturally do if only rules on e.g. `Array{Float64, N}` were defined, which is plainly a
"primitive" array type, as AD would treat them like any other `struct`.








## A Tractable Tangent Space for Structured Linear Algebra

If you treat a `struct` which subtypes `AbstractArray` like any other `struct` would be
treated, everything works.
Just let the (co)tangent type be a `Composite`.

Often it feels more intuitive to define the cotangent type to be something that mirrors the
primal type -- for example letting the (co)tangent type of a `Diagonal` be another
`Diagonal`.
This seems to be fine some of the time, as it's often possible to specify an isomorphism
between the `Composite` and this type.

Sometimes it makes less sense though.
For example, it's unclear that representing the (co)tangent of a `Fill` with another `Fill`
makes much sense at all, semantically speaking.
Certainly you could do it, but it feels like a hack when you try to get down to what you
actually _mean_, and try to marry that in with the existing meaning that a `Fill` has.

A `Composite` doesn't suffer this problem, because its only interpretation is as a
(co)tangent for a `struct`.









## [Some Peculiar AD Behaviour](@id some_peculiar_behaviour)

This scenario has nothing to do with `frule`s or `rrule`s for structured array types.
The point is to highlight that some of the peculiarities that can arise when you treat
structured array types as structs can be found elsewhere.
We'll tie this example back in to sparse linear algebra in the next section.

Consider the following programme:
```julia
using FiniteDifferences
using ForwardDiff
using LinearAlgebra

f(x::AbstractMatrix{<:Real}) = isdiag(x) ? x[diagind(x)] : x

g = sum ∘ f
X = diagm(randn(3))
```
The function `g` produces the same answer regardless which branch is taken in `f`.

FiniteDifferences.jl successfully computes the gradient of `g` at `X`:
```julia
julia> FiniteDifferences.grad(central_fdm(5, 1), g, X)[1]
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```
This seems correct.

However, ForwardDiff does not produce the same result:
```julia
julia> ForwardDiff.gradient(g, X)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```
This discrepancy occurs because `ForwardDiff` only ever hits the first branch in `f`.
On the other hand, `FiniteDifferences` will hit the second branch in `f` for any
non-diagonal perturbation of the input.
This property seems to arise from AD considering only considering a function at a single
point, rather than the neighbourhood of a point.

We've not used any contraversial `frule`s here, so this property must be something to do
with AD's semantics, not incorrect rule implementation.







### What About Reverse-Mode?

Unsurprisingly, ReverseDiff.jl and Zygote.jl both behave in the same way as ForwardDiff.jl:
```
using ReverseDiff
using Zygote

julia> ReverseDiff.gradient(g, X)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> Zygote.gradient(g, X)[1]
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```
Again, this seems to be something inherent in the way that AD handles branches.









### What Does This Have To Do With Sparse Arrays?

Consider a small modification to the previous programme:
```julia
using LinearAlgebra

f(x::AbstractMatrix{<:Real}) = isdiag(x) ? Diagonal(diag(x)) : x

g = sum ∘ f
```
We now return a `Diagonal` rather than a `Vector`, but `g` is essentially unchanged beyond
this implementation detail.
All four tools provide (essentially) the same answer as before:
```julia
using FiniteDifferences
using ForwardDiff
using ReverseDiff
using Zygote

julia> FiniteDifferences.grad(central_fdm(5, 1), g, X)[1]
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> ForwardDiff.gradient(g, X)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> ReverseDiff.gradient(g, X)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> Zygote.gradient(g, X)[1]
3×3 Diagonal{Float64, FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}}:
 1.0   ⋅    ⋅
  ⋅   1.0   ⋅
  ⋅    ⋅   1.0
```
`Zygote` has hit a couple of `rrule`s, which are worth diving into for sanity's sake.
The `rrule` for `diag` is [defined](https://github.com/JuliaDiff/ChainRules.jl/blob/76ef95c326e773c6c7140fb56eb2fd16a2af468b/src/rulesets/LinearAlgebra/structured.jl#L46)
as:
```julia
function rrule(::typeof(diag), A::AbstractMatrix)
    function diag_pullback(ȳ)
        return (NO_FIELDS, Diagonal(ȳ))
    end
    return diag(A), diag_pullback
end
```
When the input `A` is a `Matrix{Float64}` and the cotangent `ȳ` a `Vector{Float64}`, it
seems uncontraversial.

The `rrule` for the constructor for `Diagonal` is defined [here](https://github.com/JuliaDiff/ChainRules.jl/blob/76ef95c326e773c6c7140fb56eb2fd16a2af468b/src/rulesets/LinearAlgebra/structured.jl#L32)
and is similarly uncontraversial:
```julia
function rrule(::Type{<:Diagonal}, d::AbstractVector)
    function Diagonal_pullback(ȳ::AbstractMatrix)
        return (NO_FIELDS, diag(ȳ))
    end
    return Diagonal(d), Diagonal_pullback
end
```

The point of highlighting these two `rrule`s is to show that they both seem
sensible -- certainly it's unclear how else one might reasonably implement them.
The discrepancy between the usual definition of the gradient of `g` and what AD produces
must, therefore, be "AD-related" not "rule-related".







### The `sparse` Function

Recall that the function `sparse` returns a `SparseMatrixCSC` when applied to a `Matrix`,
in which only the non-zero elements of the `Matrix` are explicitly represented.
If we define `g` as follows:
```julia
using SparseArrays

g = sum ∘ sparse
X = [1.0 0 0; 1.0 0 2.0; 0 3.0 0]
```
we have a situation which is analogous to the previous `Diagonal` example.
The precise sparsity pattern in `g(X)` will depend upon the precise values of `X`.
As before, `FiniteDifferences` gets the answer you would expect, while the AD tools give
something which reflects the sparsity pattern.
```julia
julia> FiniteDifferences.grad(central_fdm(5, 1), g, X)[1]
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> ForwardDiff.gradient(g, X)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 1.0  0.0  1.0
 0.0  1.0  0.0

julia> ReverseDiff.gradient(g, X)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 1.0  0.0  1.0
 0.0  1.0  0.0
```
Zygote doesn't work on this example at the time of writing, hence its exclusion.

As before, neither `ForwardDiff` nor `ReverseDiff` know anything about the function
`sparse`, nor the `SparseMatrixCSC` type.






### Takeaway

There are a at least two possible things you could think to do as a consequence of this.
The first is to accept that this is just a feature of AD that sparse linear algebra should
inherit.
The second is to attempt to alter it using carefully chosen rules for linear algebra.

The second option appears to necessitate the same kind of dense (co)tangents as
[before](@ref functions_of_sparse_arrays), precluding differentiation of a
[very large class of important functions](@ref intermediate_functions).
Conversely, the above examples are somewhat contrived, and it's not clear whether important
examples exist in practice.
However, it doesn't resolve the underlying problem, which is AD-related, not rule-related.

The first option does leave us with some behaviour which it is hard to argue is desirable,
however, all ADs of which we are aware suffer this problem this exact problem regardless
your rule system.







## Conclusion

It seems clear that the price to pay for AD whose performance is acceptable for sparse array
types is seemingly-unintuitive results.
However, these results are easily understood if sparse arrays are simply thought of as
`struct`s, rather than array-like things, in the context of AD.

Indeed, not treating sparse arrays like other `struct`s would mean that they're inconsistent
with other `struct`s.
When viewed in this light, it is less obvious that these results are undesirable.

It does not appear, however, that we need lose any functionality by imposing such
restrictions (sparse arrays can always be `collect`ed if necessary).
