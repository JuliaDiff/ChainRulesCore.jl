# Design Notes: Why can you change the primal?
#TODO generalize intro/title to ask why `rrule` is how it is, and in particular about changing primal

These design notes are to help you understand why ChainRules allows the primal computation to be changed.
We will focus this discussion on reverse mode and `rrule`, though the same also applies to forwards mode and `frule`.
In fact, it has a particular use in forward mode for efficiently calculating the pushforward of a differential equation solve via expanding the system of equations to also include the derivatives, and solving all at once.
In forward mode it is related to the fusing of `frule` and `pushforward`.
In reverse mode we can focus on the the distinct primal and gradient passes.


## The Journey to `rrule`

Let's imagine a different system for rules, one that doesn't let you do this.
This system is what a lot of AD systems have --- it is what [Nabla.jl](https://github.com/invenia/Nabla.jl/)[^1] had originally.
We will have a primal (i.e. forward) pass that directly executes the primal function and just records its _inputs_ and its _output_ (as well as the _primal function_ itself) onto the tape.[^2].
Then during the gradient (i.e. reverse) pass it has a function which receives those records from the tape along with the sensitivity of the output, and gives back the sensitivity of the input.
We will call this function `pullback_at`, as it pulls back the sensitivity at a given primal point.
To make this concrete:
```julia
y = f(x)  # primal program
x̄ = pullback_at(f, x, y, ȳ)
```

Let's write one:
```julia
y = sin(x)
pullback_at(::typeof(sin), x, y, ȳ) = ȳ * cos(x)
```

Great. So far so good.
As a short exercise the reader might like to implement the one for the [logistic sigmoid](https://en.wikipedia.org/wiki/Logistic_function#Derivative).
It also works without issue.
#TODO should I actually just include logistic sigmoid a second example though out, and write its code next the the code for `sin`?


Now let's consider why we implement `rrules` like this in the first place.
One key reason, [^3] is to allow us to insert our domain knowledge to do better than the AD would do just by breaking everything down into `+` and `*` etc.
What insights do we have about `sin` and `cos`?
Here is one:
```julia
julia> @btime sin(x) setup=(x=rand());
  3.838 ns (0 allocations: 0 bytes)

julia> @btime cos(x) setup=(x=rand());
  4.795 ns (0 allocations: 0 bytes)

julia> 3.838 + 4.795
8.633

julia> @btime sincos(x) setup=(x=rand());
  6.028 ns (0 allocations: 0 bytes)
```
It is \~30%[^4] faster to compute `sin` and `cos` at the same time via `sincos` than it is to compute them one after the other.
How can we incorporate this insight into our system?
We know that we can compute the `cos(x)` at the same time as the `sin(x)` is computed in primal, because it only depends on `x` --- we don't need to know `ȳ`.
But there is nowhere to put it that is accessible both to the primal pass and the gradient pass code.

What if we introduced some variable called `intermediates` that is also recorded onto the tape during the primal pass?
We would need to be able to modify the primal pass to do this, so that we can actually put the data into the `intermediates`.
So we will introduce a function: `augmented_primal`, that will return the primal output plus the `intermediates` that we want to reuse in the gradient pass.
Then we will make our AD system replace calls to the primal with calls to the `augmented_primal` of the primal function and take care of all the bookkeeping.
So that would look like:
```julia
y = f(x)  # primal program
y, intermediates = augmented_primal(f, x)
x̄ = pullback_at(f, x, y, ȳ, intermediates)
```

Let's try writing this now:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, (; cx=cx)  # use a NamedTuple for the intermediates
end

pullback_at(::typeof(sin), x, y, ȳ, intermediates) = ȳ * intermediates.cx
```

Cool!
That lets us do what we wanted.
We net decreased the time it takes to run the primal and gradient passes.
We have now demonstrated the title question of why we needed to be able to modify the primal pass.
We will go into that more later, and have some more examples of use.
But first let's continue to see how we go from that `augmented_primal` to `pullback_at` to [`rrule`](@ref).


One thing we notice when looking at `pullback_at` is it really is starting to have a lot of arguments.
It had a fair few already, and now we are adding `intermediates` as well.
Not to mention this is a fairly simple function, only 1 input, no keyword arguments.
Furthermore, we don't even use all of them all the time.
The original new code for pulling back `sin` no longer needs the `x`, and it never needed `y` (though `sigmoid` does).
Having all this extra stuff means that the `pullback_at` function signature is long, and we are putting a bunch of extra stuff on the tape, using more memory.
What if we generalized the idea of the `intermediate` named tuple, and had a struct that just held anything we might want put on the tape.
```julia
struct PullbackMemory{P, S}
  primal_function::P
  state::S
end
# convenience constructor:
Memory(primal_function; state...) = PullbackMemory(primal_function, state)
# convenience accessor so that `m.x` is same as `m.state.x`
Base.getproperty(m::PullbackMemory, propname) = getproperty(getfield(m, :state), propname)
```

So changing our API we have:
```julia
y = f(x)  # primal program
y, pb = augmented_primal(f, x)
x̄ = pullback_at(pb, ȳ)
```
which is much cleaner.

Let's try writing with this now:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end

pullback_at(pb::PullbackMemory{typeof(sin)}, ȳ) = ȳ * pb.cx
```

I think that looks pretty nice.

One way we could make it look a bit nicer for usage is if the `PullbackMemory` was actually a callable object,
since `pullback_at` only has the 2 arguments,
and conceptually the `PullbackMemory` is a more fixed thing -- it is fully determined by the end of the primal pass,
then during the reverse pass the argument `ȳ` is successively computed.
We can do that via call overloading:
```julia
y = f(x)  # primal program
y, pb = augmented_primal(f, x)
x̄ = pb(ȳ)
```
and for `sin`:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end
(pb::PullbackMemory)(ȳ) = ȳ * pb.cx
```

Those looking closely will spot what we have done here.
We now have a object (`pb`) that acts on the cotangent of the output of the primal (`ȳ`) to give us the cotangent of the input of the primal function (`x̄`).
_`pb` is not just the **memory** of state required for the `pullback`, it **is** the pullback._

We have one final thing to do.
Lets think about make the code easy to modify.
Lets go back and think about the changes we would have to make to swap to using `sincos` from our original `sin` then `cos` way of writing.
To rewrite that original formulation in the new pullback form we have:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y = sin(x)
  return y, PullbackMemory(sin; x=x)
end
(pb::PullbackMemory)(ȳ) = ȳ * cos(pb.x)
```
To go from that to:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end
(pb::PullbackMemory)(ȳ) = ȳ * pb.cx
```
we need to make a series of changes.
We need to update what work is done in the primal to compute `cx`.
We need to update what was stored in the `PullbackMemory`.
And we need to update the the function that applies the pullback so it uses the new thing that was stored.
It's important these parts all stay in sync.
It's not too bad for this simple example with just one thing to remember.
For more complicated multi-argument functions, like will be talked about below, you often end up needing to remember half a dozen things, like sizes and indices relating to each input/output.
So it gets a little more fiddly to make sure you remember all the things you need to a and give them same name in both places.
_Is there a way we can automatically just have all the things we use remembered for us?_

Surprisingly for such a specific request, there actually is.
This is a closure.
A closure in julia is a callable structure, that automatically contains a field for every object from its parent scope that is used in its body.
There are [incredible ways to abuse this](https://invenia.github.io/blog/2019/10/30/julialang-features-part-1#closures-give-us-classic-object-oriented-programming); but here we can in-fact use closures exactly as they are intended.
Replacing `PullbackMemory` with a closure that works the same way lets us avoid having to manually control what is remembered, _and_ lets us avoid separately writing the call overload.
So we have for `sin`:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  pb = ȳ -> cx * ȳ  # pullback closure. closes over `cx`
  return y, pb
end
```

Our `augmented_primal` is now with-in spitting distance of `rrule`.
All that is left is a rename, and some extra conventions around multiple outputs and gradients with respect to callable objects.

This has been a journey into how we get to [`rrule`](@ref) as it is defined in `ChainRulesCore`.
We started with an unaugmented primal function and a `pullback_at` function that only saw the inputs and outputs of the primal.
We realized a key limitation of this was that we couldn't share computational work between the primal and and gradient passes.
To solve this we introduced the notation of a some `intermediate` that is shared from the primal to the pullback.
We successively improved that idea, first by making it a type that held everything that is needed for the pullback: the `PullbackMemory`.
Which we then made callable --- so it  was itself the pullback.
Finally, we replaced that separate callable structure with a closure, which kept everything in one place and made it more convenient.

## More Shared Work Examples
`sincos` is a nice simple example of when it is useful to share work between primal and the pullback.
There are many others though.
It is surprising that in so many cases it is reasonable to write the rules where the only shared information between the primal and the pullback is the primal inputs, or primal outputs.
Under our formulation above, those primal inputs/outputs are shared information just like any other.
Beyond this there are a number other decent applications.

### `getindex`
In julia (and many other numerical languages) indexing can take many more arguments than simply a couple of integers.
For example, passing in boolean masking arrays (logical indexing), passing in ranges to do slices, etc.
Converting them to plain integers, arrays of integers, and ranges is `Base.to_indices` is the first thing that `getindex` does.
It then re-calls `getindex` with these simpler types to get the result.

The result of pulling back the `getindex` operation is always an array that is all zeros,
except for the elements that are selected, which are set to the appropriate sensitivities being pulled back.
So identify which actual positions in the array are being gotten/set is common work to both primal and gradient computations.
We really don't want to deal with fancy indexing types during the pullback, because there are weird edge cases like indexing in such a way that the same element is output twice (and thus we have 2 sensitivities we need to add to it).
We can pull the `to_indices` out of the primal computation and remember the plain indexes used, then can reuse them to set gradients during the pullback.

See the [code for this in ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/Base/indexing.jl)

### `exp(::Matrix)`
[Matrix Functions](https://en.wikipedia.org/wiki/Matrix_function) are functions that are generalized to operate on matrixes rather than scalars.
Note that this is distinct from simply element-wise application of the function to the matrix's elements.
[Matrix Exponential](https://en.wikipedia.org/wiki/Matrix_exponential) `exp(::Matrix)` is one particularly important matrix function.


[Al-Mohy, Awad H. and Higham, Nicholas J. (2009) _Computing the Fréchet Derivative of the Matrix Exponential, with an application to Condition Number Estimation_. SIAM Journal On Matrix Analysis and Applications., 30 (4). pp. 1639-1657. ISSN 1095-7162](http://eprints.maths.manchester.ac.uk/1218/), published a method for this.
It is pretty complex and very cool.
To quote its abstract (emphasis mine):

> The algorithm is derived from the scaling and squaring method by differentiating the Padé approximants and the squaring recurrence, **re-using quantities computed during the evaluation of the Padé approximant**, and intertwining the recurrences in the squaring phase.

Julia does in fact use a Padé approximation to compute `exp(::Matrix)`.
So we can extract the code for that into our augmented primal, and add remembering the intermediate quantities that are to be used.
See the [code for this in ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/LinearAlgebra/matfun.jl)


An interesting thing here that may be of concern to some:
if Julia changes the algorithm it it uses to compute `exp(::Matrix)` then during an AD primal pass, it will continue to use the old Padé approximation based algorithm.
This is a real thing that might happen, there are many other algorithms that can compute the matrix exponential.
Further, perhaps there might be an improvement to the exact coefficient or cut-offs used by julia's current Padé approximation.
This is not a breaking change on Julia's behalf: [exact floating point numerical values are not generally considered part of the SemVer-bound API](http://colprac.sciml.ai/#changes-that-are-not-considered-breaking).
Rather only the general accuracy of the computed value relative to the true mathematical value (e.g. for common scalar operations Julia promises 1ULP).

This change will result in the output of the AD primal pass not being exactly equal to that that would be seen from just running the primal code.
It will still be accurate because the current implementation is accurate, but it will be different.
It is [our argument](https://forums.swift.org/t/agreement-of-valuewithdifferential-and-value/31869) that in general this should be considered acceptable.
As long as it is in general about as accurate as the unaugmented primal it should be acceptable.
E.g. it might overshoot for some values the unaugmented primal undershoots for.


### `eigvals`
`eigvals` is a real case where the algorithm for the augmented primal, and the original primal _is already different today_.
To compute the pullback of `eigvals` you need to know not only the eigenvalues, but also the eigenvectors.
the `eigen` function computes both, so that is used in augmented primal.
See the [code for this in ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/LinearAlgebra/factorization.jl#L209-L218).
If we could not compute remember the eigenvectors in the primal pass we would have to call `eigen` in the gradient pass anyway, and fully recomputed eigenvectors and eigenvalues; over doubling the total work.

However, if you trace this down, it actually uses a different algorithm.

`eigvals` basically wraps `LAPACK.syevr!('N', ...)`
which goes through [DSYEVR](http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_gaeed8a131adf56eaa2a9e5b1e0cce5718.html)
and eventually calls [DSTERF](http://www.netlib.org/lapack/explore-html/d2/d24/group__aux_o_t_h_e_rcomputational_gaf0616552c11358ae8298d0ac18ac023c.html#gaf0616552c11358ae8298d0ac18ac023c)
which uses _"Pal-Walker-Kahan variant of the QL or QR algorithm."_ to compute eigenvalues

In contrast, `eigen` wraps `LAPACK.syevr!('V',...)`
which also goes through [DSYEVR](http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_gaeed8a131adf56eaa2a9e5b1e0cce5718.html)
but eventually calls [DSTEMR](http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga14daa3ac4e7b5d3712244f54ce40cc92.html#ga14daa3ac4e7b5d3712244f54ce40cc92)
which calculates eigenvalues _"either by bisection or the dqds algorithm."_.

Both of these are very good algorithms.
LAPACK has had decades of work by experts and is one of the most trusted libraries for linear algebra.
But they are different algorithms that give different results.
The differences in practices are around $10^{-15}$, which while very small on absolute terms
are as far as `Float64` is concerned a very real difference.

### Matrix Division
Roughly speaking:
`Y=A\B` is the function that finds the least-square solution to `YA ≈ B`.
When solving such a system, the efficient way to do so is to factorize `A` into an appropriate factorized form such as `Cholesky` or `QR` etc,
then perform the `\` operation on the factorized form.
The pullback of `A\B` with respect to `B` is `Ȳ-> A' \ Ȳ`.
It should be noted that that involves computing the factorization of `A'` (the adjoint of `A`).
In this computation the factorization of the original `A` can reused.
Doing so can give a 4x speed-up.

We don't have this in ChainRules.jl yet, because Julia is missing some definitions of `adjoint` of factorizations ([JuliaLang/julia#38293](https://github.com/JuliaLang/julia/issues/38293)).
I have been promised them for Julia v1.7 though.
You can see what the code would look like in [PR #302](https://github.com/JuliaDiff/ChainRules.jl/pull/302).


# Conclusion
This document has explained why [`rrule`](@ref) is the way it is.
In particular it has highlighted why the primal computation is able to be changed from simply calling the function.
Futher, it has explained why `rrule` returns a closure for the pullback, rather than it being a seperate function.
It has highlighted several places in ChainRules.jl where this has allowed us to have significantly improved performance.
Being able to change the primal computation is practically essential for a high performance AD system.





[^1]:
    I am not just picking on Nabla randomly.
    Many of the core developers of ChainRules worked on Nabla prior.
    It's a good AD, but ChainRules incorporates lessons learned from working on Nabla.

[^2]: which may be an explicit tape, or an implicit tape that is actually incorporated into generated code (ala Zygote)

[^3]:
    Another key reason is if the operations is a primitive that is not defined in terms of more basic operations.
    In many languages this is the case for `sin`; where the actual implementation is in some separate `libm.so`.
    But actually [`sin` in Julia is defined in terms of a polynomial](https://github.com/JuliaLang/julia/blob/caeaceff8af97565334f35309d812566183ec687/base/special/trig.jl).
    It's fairly vanilla julia code.
    It shouldn't be too hard for an AD that only knows about basic operations like `+` and `*` to AD through it.
    Though that will incur something that looks a lot like truncation error (in apparent violation of Griewank and Walther's 0th Rule of AD).
    In anycase, that is another discussion, for another day.

[^4]:
    Sure, this is small-fries and depending on julia version might just get solved by the optimizer[^5], but go with it for the sake of example.

[^5]:
    To be precise this is very likely to be solved by the optimizer inlining both and then performing common subexpression elimination, with the result that it generates the code for `sincos` just form having `sin` and `cos` inside the same function.
    However, this actually doesn't apply in the case of AD as it is not possible to inline code called in the gradient pass into the primal pass -- those are seperate functions called at very different times.
    This is something [opaque closures](https://github.com/JuliaLang/julia/pull/37849) should help solve.
