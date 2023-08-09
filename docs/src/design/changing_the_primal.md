# [Design Notes: Why can you change the primal computation?](@id change_primal)

These design notes are to help you understand ChainRules.jl's [`rrule`](@ref) function.
It explains why we have a `rrule` function that returns both the primal result (i.e. the output for the forward pass) and the pullback as a closure.
It might be surprising to some AD authors, who might expect just a function that performs the pullback, that the `rrule` function computes the primal result as well as the pullback.
In particularly, `rrule` allows you to _change_ how the primal result is computed.
We will illustrate in this document why being able to change the computation of the primal is crucial for efficient AD.

!!! note "What about `frule`?"
    Discussion here is focused on on reverse mode and `rrule`.
    Similar concerns do apply to forward mode and `frule`.
    In forward mode these concerns lead to the fusing of the `pushforward` into `frule`.
    All the examples given here also apply in forward mode.
    In fact in forward mode there are even more opportunities to take advantage of sharing work between the primal and derivative computations.
    A particularly notable example is in efficiently calculating the pushforward of solving a differential equation via expanding the system of equations to also include the derivatives before solving it.

## The Journey to `rrule`

Let's imagine a different system for rules, one that doesn't let you define the computation of the primal.
This system is what a lot of AD systems have.
It is what [Nabla.jl](https://github.com/invenia/Nabla.jl/) had originally.[^1]
We will have a primal (i.e. forward) pass that directly executes the primal function and just records the primal _function_, its _inputs_ and its _output_ onto the tape.[^2].
Then during the gradient (i.e. reverse) pass it has a function which receives those records from the tape along with the sensitivity of the output, and gives back the sensitivity of the input.
We will call this function `pullback_at`, as it pulls back the sensitivity at a given primal point.
To make this concrete:
```julia
y = f(x)  # primal program
x̄ = pullback_at(f, x, y, ȳ)
```
Let's illustrate this with examples for `sin` and for the [logistic sigmoid](https://en.wikipedia.org/wiki/Logistic_function#Derivative).

```@raw html
<details open><summary>Example for `sin`</summary>
```
```julia
y = sin(x)
pullback_at(::typeof(sin), x, y, ȳ) = ȳ * cos(x)
```
`pullback_at` uses the primal input `x`, and the sensitivity being pulled back `ȳ`.

```@raw html
</details>
```
```@raw html
<details open><summary>Example for the logistic sigmoid</summary>
```

```julia
σ(x) = 1/(1 + exp(-x))  # = exp(x) / (1 + exp(x))
y = σ(x)
pullback_at(::typeof(σ), x, y, ȳ) = ȳ * y * σ(-x)  # = ȳ * σ(x) * σ(-x)
```
Notice that in `pullback_at` we are not only using input `x` but also using the primal output `y` .
This is a nice bit of symmetry that shows up around `exp`.
```@raw html
</details>
```

Now let's consider why we implement `rrule`s in the first place.
One key reason is to insert domain knowledge so as to compute the derivative more efficiently than AD would just by breaking everything down into `+`, `*`, etc.[^3]
What insights do we have about `sin` and `cos`?
What about using `sincos`?
```@raw html
<details open><summary>Example for `sin`</summary>
```
```julia
julia> using BenchmarkTools

julia> @btime sin(x) setup=(x=rand());
  3.838 ns (0 allocations: 0 bytes)

julia> @btime cos(x) setup=(x=rand());
  4.795 ns (0 allocations: 0 bytes)

julia> 3.838 + 4.795
8.633
```
vs computing both together:
```julia
julia> @btime sincos(x) setup=(x=rand());
  6.028 ns (0 allocations: 0 bytes)
```
```@raw html
</details>
```

What about the logistic sigmoid?
We note that the two values we need are `σ(x)` and `σ(-x)`
If we write these as:
$\sigma(x) = \frac{e^x}{1+e^x}$ and
$\sigma(-x) = \frac{1}{1+e^x}$
then we see they have the common term $e^x$.
`exp(x)` is a much more expensive operation than `+` and `/`.
So we can save time, if we can reuse that `exp(x)`.
```@raw html
<details open><summary>Example for the logistic sigmoid</summary>
```
If we have to computing separately:
```julia
julia> @btime 1/(1+exp(x)) setup=(x=rand());
  5.622 ns (0 allocations: 0 bytes)

julia> @btime 1/(1+exp(-x)) setup=(x=rand());
  6.036 ns (0 allocations: 0 bytes)

julia> 5.622 + 6.036
11.658
```

vs reusing `exp(x)`:
```julia
julia> @btime exp(x) setup=(x=rand());
  5.367 ns (0 allocations: 0 bytes)

julia> @btime ex/(1+ex) setup=(ex=exp(rand()));
  1.255 ns (0 allocations: 0 bytes)

julia> @btime 1/(1+ex) setup=(ex=exp(rand()));
  1.256 ns (0 allocations: 0 bytes)

julia> 5.367 + 1.255 + 1.256
7.878
```
```@raw html
</details>
```

So we are talking about a 30-40% speed-up from these optimizations.[^4]

It is faster to  compute `sin` and `cos` at the same time via `sincos` than it is to compute them one after the other.
And it is faster to reuse the `exp(x)` in computing `σ(x)` and `σ(-x)`.
How can we incorporate this insight into our system?
We know we can compute both of these in the primal — because they only depend on `x` and not on `ȳ` — but there is nowhere to put them that is accessible both to the primal pass and the gradient pass code.

What if we introduced some variable called `intermediates` that is also recorded onto the tape during the primal pass?
We would need to be able to modify the primal pass to do this, so that we can actually put the data into the `intermediates`.
So we will introduce a function: `augmented_primal`, that will return the primal output plus the `intermediates` that we want to reuse in the gradient pass.
Then we will make our AD system replace calls to the primal with calls to the `augmented_primal` of the primal function and take care of all the bookkeeping.
So that would look like:
```julia
y = f(x)  # primal program
y, intermediates = augmented_primal(f, x)
x̄ = pullback_at(f, x, y, ȳ, intermediates)
```

```@raw html
<details open><summary>Example for `sin`</summary>
```
```julia
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, (; cx=cx)  # use a NamedTuple for the intermediates
end

pullback_at(::typeof(sin), x, y, ȳ, intermediates) = ȳ * intermediates.cx
```
```@raw html
</details>
```

```@raw html
<details open><summary>Example for the logistic sigmoid</summary>
```
```julia
function augmented_primal(::typeof(σ), x)
  ex = exp(x)
  y = ex / (1 + ex)
  return y, (; ex=ex)  # use a NamedTuple for the intermediates
end

pullback_at(::typeof(σ), x, y, ȳ, intermediates) = ȳ * y / (1 + intermediates.ex)
```
```@raw html
</details>
```

Cool!
That lets us do what we wanted.
We net decreased the time it takes to run the primal and gradient passes.
We have now demonstrated the title question of why we want to be able to modify the primal pass.
We will go into that more later and have some more usage examples, but first let's continue to see how we go from `augmented_primal` and `pullback_at` to [`rrule`](@ref).

One thing we notice when looking at `pullback_at` is it really is starting to have a lot of arguments.
It had a fair few already, and now we are adding `intermediates` as well, making it even more unwieldy.
Not to mention these are fairly simple example, the `sin` and `σ` functions have 1 input and no keyword arguments.
Furthermore, we often don't even use all of the arguments to `pullback_at`.
The new code for pulling back `sin` — which uses `sincos` and `intermediates` — no longer needs `x`, and it never needed `y` (though sigmoid `σ` does).
And storing all these things on the tape — inputs, outputs, sensitivities, intermediates — is using up extra memory.
What if we generalized the idea of the `intermediate` named tuple, and had `augmented_primal` return a struct that just held anything we might want put on the tape.
```julia
struct PullbackMemory{P, S}
  primal_function::P
  state::S
end
# convenience constructor:
PullbackMemory(primal_function; state...) = PullbackMemory(primal_function, state)
# convenience accessor so that `m.x` is same as `m.state.x`
Base.getproperty(m::PullbackMemory, propname) = getproperty(getfield(m, :state), propname)
```

So changing our API we have:
```julia
y = f(x)  # primal program
y, pb = augmented_primal(f, x)
x̄ = pullback_at(pb, ȳ)
```
which is much cleaner.

```@raw html
<details open><summary>Example for `sin`</summary>
```
```julia
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end

pullback_at(pb::PullbackMemory{typeof(sin)}, ȳ) = ȳ * pb.cx
```
```@raw html
</details>
```

```@raw html
<details open><summary>Example for the logistic sigmoid</summary>
```
```julia
function augmented_primal(::typeof(σ), x)
  ex = exp(x)
  y = ex / (1 + ex)
  return y, PullbackMemory(σ; y=y, ex=ex)
end

pullback_at(pb::PullbackMemory{typeof(σ)}, ȳ) = ȳ * pb.y / (1 + pb.ex)
```
```@raw html
</details>
```

That now looks much simpler; `pullback_at` only ever has 2 arguments.

One way we could make it nicer to use is by making `PullbackMemory` a callable object.
Conceptually, for a particular evaluation of an operation, the `PullbackMemory` is fixed.
It is fully determined by the end of the primal pass.
The during the gradient (reverse) pass the `PullbackMemory` is used to successively compute the `ȳ`  argument.
So it makes sense to make `PullbackMemory` a callable object that acts on the sensitivity.
We can do that via call overloading:
```julia
y = f(x)  # primal program
y, pb = augmented_primal(f, x)
x̄ = pb(ȳ)
```

```@raw html
<details open><summary>Example for `sin`</summary>
```
```julia
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end
(pb::PullbackMemory{typeof(sin)})(ȳ) = ȳ * pb.cx
```

```@raw html
</details>
```

```@raw html
<details open><summary>Example for the logistic sigmoid</summary>
```
```julia
function augmented_primal(::typeof(σ), x)
  ex = exp(x)
  y = ex / (1 + ex)
  return y, PullbackMemory(σ; y=y, ex=ex)
end

(pb::PullbackMemory{typeof(σ)})(ȳ) = ȳ * pb.y / (1 + pb.ex)
```
```@raw html
</details>
```

Let's recap what we have done here.
We now have an object `pb` that acts on the cotangent of the output of the primal `ȳ` to give us the cotangent of the input of the primal function `x̄`.
_`pb` is not just the **memory** of state required for the `pullback`, it **is** the pullback._

We have one final thing to do, which is to think about how we make the code easy to modify.
Let's go back and think about the changes we would have make to go from our original way of writing that only used the inputs/outputs, to one that used the intermediate state.

```@raw html
<details open><summary>Example for `sin`</summary>
```
To rewrite that original formulation in the new pullback form we have:
```julia
function augmented_primal(::typeof(sin), x)
  y = sin(x)
  return y, PullbackMemory(sin; x=x)
end
(pb::PullbackMemory)(ȳ) = ȳ * cos(pb.x)
```
To go from that to:
```julia
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end
(pb::PullbackMemory)(ȳ) = ȳ * pb.cx
```
```@raw html
</details>
```

```@raw html
<details open><summary>Example for the logistic sigmoid</summary>
```
```julia
function augmented_primal(::typeof(σ), x)
  y = σ(x)
  return y, PullbackMemory(σ; y=y, x=x)
end
(pb::PullbackMemory{typeof(σ)})(ȳ) = ȳ * pb.y * σ(-pb.x)
```
to get to:
```julia
function augmented_primal(::typeof(σ), x)
  ex = exp(x)
  y = ex/(1 + ex)
  return y, PullbackMemory(σ; y=y, ex=ex)
end
(pb::PullbackMemory{typeof(σ)})(ȳ) = ȳ * pb.y/(1 + pb.ex)
```
```@raw html
</details>
```

We should think about how we might want to make future changes to this code.[^6]

We need to make a series of changes:
 * update what work is done in the primal, to compute the intermediate values.
 * update what is stored in the `PullbackMemory`.
 * update the function that applies the pullback so it uses the new thing that was stored.
It's important these parts all stay in sync.
It's not too bad for this simple example with just one or two things to remember.
For more complicated multi-argument functions, which we will show below, you often end up needing to remember half a dozen things, like sizes and indices relating to each input/output, so it gets a little more fiddly to make sure you remember all the things you need to and give them the same name in both places.
_Is there a way we can automatically just have all the things we use remembered for us?_ 
Surprisingly for such a specific request, there actually is: a closure.

A closure in Julia is a callable structure that automatically contains a field for every object from its parent scope that is used in its body.
There are [incredible ways to abuse this](https://invenia.github.io/blog/2019/10/30/julialang-features-part-1#closures-give-us-classic-object-oriented-programming); but here we can use closures exactly as they are intended.
Replacing `PullbackMemory` with a closure that works the same way lets us avoid having to manually control what is remembered _and_ lets us avoid separately writing the call overload.

```@raw html
<details open><summary>Example for `sin`</summary>
```
```julia
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  pb = ȳ -> cx * ȳ  # pullback closure. closes over `cx`
  return y, pb
end
```
```@raw html
</details>
```

```@raw html
<details open><summary>Example for the logistic sigmoid</summary>
```
```julia
function augmented_primal(::typeof(σ), x)
  ex = exp(x)
  y = ex / (1 + ex)
  pb = ȳ -> ȳ * y / (1 + ex)  # pullback closure. closes over `y` and `ex`
  return y, pb
end
```
```@raw html
</details>
```
This is pretty clean now.

Our `augmented_primal` is now within spitting distance of `rrule`.
All that is left is a rename and some extra conventions around multiple outputs and gradients with respect to callable objects.

This has been a journey into how we get to [`rrule`](@ref) as it is defined in `ChainRulesCore`.
We started with an unaugmented primal function and a `pullback_at` function that only saw the inputs and outputs of the primal.
We realized a key limitation of this was that we couldn't share computational work between the primal and gradient passes.
To solve this we introduced the notation of some `intermediate` that is shared from the primal to the pullback.
We successively improved that idea, first by making it a type that held everything that is needed for the pullback: the `PullbackMemory`, which we then made callable, so it was itself the pullback.
Finally, we replaced that separate callable structure with a closure, which kept everything in one place and made it more convenient.

## More Shared Work Examples
`sin` and the logistic sigmoid are nice, simple examples of when it is useful to share work between the primal and the pullback.
There are many others though.
It is actually surprising that in so many cases it is reasonable to write the rules where the only shared information between the primal and the pullback is the primal inputs (like our original `sin`), or primal outputs (like our original logistic sigmoid).
Under our formulation above, those primal inputs/outputs are shared information just like any other.
Beyond this, there are a number of other decent applications.

### `getindex`
In Julia (and many other numerical languages) indexing can take many more arguments than simply a couple of integers, such as boolean masking arrays (logical indexing), ranges for slices, etc.
Converting the arguments to plain integers, arrays of integers, and ranges with `Base.to_indices` is the first thing that `getindex` does.
It then re-calls `getindex` with these simpler types to get the result.

The result of pulling back the `getindex` operation is always an array that is all zeros, except for the elements that are selected, which are set to the appropriate sensitivities being pulled back.
To identify which actual positions in the array are being gotten/set is common work to both primal and gradient computations.
We really don't want to deal with fancy indexing types during the pullback, because there are weird edge cases like indexing in such a way that the same element is output twice (and thus we have 2 sensitivities we need to add to it).
We can pull the `to_indices` out of the primal computation and remember the plain indexes used, then can reuse them to set gradients during the pullback.

See the [code for this in ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/Base/indexing.jl)

### `exp(::Matrix)`
[Matrix Functions](https://en.wikipedia.org/wiki/Matrix_function) are generalizations of scalar functions to operate on matrices.
Note that this is distinct from simply element-wise application of the function to the matrix's elements.
The [Matrix Exponential](https://en.wikipedia.org/wiki/Matrix_exponential) `exp(::Matrix)` is a particularly important matrix function.

Al-Mohy and Higham (2009)[^7], published a method for computing the pullback of `exp(::Matrix)`.
It is pretty complex and very cool.
To quote its abstract (emphasis mine):

> The algorithm is derived from the scaling and squaring method by differentiating the Padé approximants and the squaring recurrence, **re-using quantities computed during the evaluation of the Padé approximant**, and intertwining the recurrences in the squaring phase.

Julia does in fact use a Padé approximation to compute `exp(::Matrix)`.
So we can extract the code for that into our augmented primal, and add remembering the intermediate quantities that are to be used.
See the [code for this in ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/LinearAlgebra/matfun.jl)

An interesting scenario here that may be of concern to some:
if Julia changes the algorithm it uses to compute `exp(::Matrix)`, then during an AD primal pass, it will continue to use the old Padé approximation based algorithm.
This may actually happen, as there are many other algorithms that can compute the matrix exponential.
Further, perhaps there might be an improvement to the exact coefficient or cut-offs used by Julia's current Padé approximation.
If Julia made this change it would not be considered breaking.
[Exact floating point numerical values are not generally considered part of the SemVer-bound API](https://docs.sciml.ai/ColPrac/stable/#Changes-that-are-not-considered-breaking).
Rather only the general accuracy of the computed value relative to the true mathematical value (e.g. for common scalar operations Julia promises 1 [ULP](https://en.wikipedia.org/wiki/Unit_in_the_last_place)).

This change will result in the output of the AD primal pass not being exactly equal to what would be seen from just running the primal code.
It will still be accurate because the current implementation is accurate, but it will be different.
It is [our argument](https://forums.swift.org/t/agreement-of-valuewithdifferential-and-value/31869) that in general this should be considered acceptable, as long as the AD primal pass is in general about as accurate as the unaugmented primal.
E.g. it might overshoot for some values the unaugmented primal undershoots for.

### `eigvals`
`eigvals` is a real case where the algorithm for the augmented primal and the original primal _is already different today_.
To compute the pullback of `eigvals` you need to know not only the eigenvalues but also the eigenvectors.
The `eigen` function computes both, so that is used in the augmented primal.
See the [code for this in ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/LinearAlgebra/factorization.jl#L209-L218).
If we could not compute and remember the eigenvectors in the primal pass, we would have to call `eigen` in the gradient pass anyway and fully recompute eigenvectors and eigenvalues, more than doubling the total work.

However, if you trace this down, it actually uses a different algorithm.

`eigvals` basically wraps `LAPACK.syevr!('N', ...)`, which goes through [DSYEVR](http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_gaeed8a131adf56eaa2a9e5b1e0cce5718.html) and eventually calls [DSTERF](http://www.netlib.org/lapack/explore-html/d2/d24/group__aux_o_t_h_e_rcomputational_gaf0616552c11358ae8298d0ac18ac023c.html#gaf0616552c11358ae8298d0ac18ac023c), which uses _"Pal-Walker-Kahan variant of the QL or QR algorithm."_ to compute eigenvalues

In contrast, `eigen` wraps `LAPACK.syevr!('V',...)` which also goes through [DSYEVR](http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_gaeed8a131adf56eaa2a9e5b1e0cce5718.html) but eventually calls [DSTEMR](http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga14daa3ac4e7b5d3712244f54ce40cc92.html#ga14daa3ac4e7b5d3712244f54ce40cc92), which calculates eigenvalues _"either by bisection or the dqds algorithm."_.

Both of these are very good algorithms.
LAPACK has had decades of work by experts and is one of the most trusted libraries for linear algebra.
But they are different algorithms that give different results.
The differences in practice are around $10^{-15}$, which while very small on absolute terms are as far as `Float64` is concerned a very real difference.

### Matrix Division
Roughly speaking:
`Y=A\B` is the function that finds the least-square solution to `YA ≈ B`.
When solving such a system, the efficient way to do so is to factorize `A` into an appropriate factorized form such as `Cholesky` or `QR`, then perform the `\` operation on the factorized form.
The pullback of `A\B` with respect to `B` is `Ȳ -> A' \ Ȳ`.
It should be noted that this involves computing the factorization of `A'` (the adjoint of `A`).[^8]
In this computation the factorization of the original `A` can reused.
Doing so can give a 4x speed-up.

We don't have this in ChainRules.jl yet, because Julia is missing some definitions of `adjoint` of factorizations ([JuliaLang/julia#38293](https://github.com/JuliaLang/julia/issues/38293)).[^8]
We have been promised them for Julia v1.7 though.
You can see what the code would look like in [PR #302](https://github.com/JuliaDiff/ChainRules.jl/pull/302).

## Conclusion
This document has explained why [`rrule`](@ref) is the way it is.
In particular it has highlighted why the primal computation is able to be changed from simply calling the function.
Further, it has explained why `rrule` returns a closure for the pullback, rather than it being a separate function.
It has highlighted several places in ChainRules.jl where this has allowed us to significantly improve performance.
Being able to change the primal computation is practically essential for a high performance AD system.

[^1]:
    I am not just picking on Nabla randomly.
    Many of the core developers of ChainRules worked on Nabla prior.
    It's a good AD, but ChainRules incorporates lessons learned from working on Nabla.

[^2]: which may be an explicit tape or an implicit tape that is actually incorporated into generated code (à la Zygote)

[^3]:
    Another key reason is if the operation is a primitive that is not defined in terms of more basic operations.
    In many languages this is the case for `sin`; where the actual implementation is in some separate `libm.so`.
    But actually `sin` in Julia is [defined in terms of a polynomial](https://github.com/JuliaLang/julia/blob/caeaceff8af97565334f35309d812566183ec687/base/special/trig.jl).
    It's fairly vanilla Julia code.
    It shouldn't be too hard for an AD that only knows about basic operations like `+` and `*` to AD through it.
    In any case, that is another discussion for another day.

[^4]:
    Sure, this is small fries and depending on Julia version might just get solved by the optimizer[^5], but go with it for the sake of example.

[^5]:
    To be precise, this is very likely to be solved by the optimizer inlining both and then performing common subexpression elimination, with the result that it generates the code for `sincos` just from having `sin` and `cos` inside the same function.
    However, this actually doesn't apply in the case of AD, as it is not possible to inline code called in the gradient pass into the primal pass.
    Those are separate functions called at very different times.
    This is something [opaque closures](https://github.com/JuliaLang/julia/pull/37849) should help solve.

[^6]:
    One change we might consider is to have logistic sigmoid to only remember one thing.
    Rather than remembering `y` and `ex` to use in the pullback, we could compute `y / (1 + ex)` during the augmented primal, and just remember that.

[^7]: [Al-Mohy, Awad H. and Higham, Nicholas J. (2009) _Computing the Fréchet Derivative of the Matrix Exponential, with an application to Condition Number Estimation_. SIAM Journal On Matrix Analysis and Applications., 30 (4). pp. 1639-1657. ISSN 1095-7162](http://eprints.maths.manchester.ac.uk/1218/)

[^8]: To be clear here we mean `adjoint` as in the conjugate transpose of a matrix, rather than in the sense of reverse mode AD.
