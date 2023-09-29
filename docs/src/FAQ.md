# FAQ

## What is up with the different symbols?

### `Δx`, `∂x`, `dx`
ChainRules uses these perhaps atypically.
As a notation that is the same across propagators, regardless of direction (in contrast see `ẋ` and `x̄` below).

 - `Δx` is the input to a propagator, (i.e a _seed_ for a _pullback_; or a _perturbation_ for a _pushforward_).
 - `∂x` is the output of a propagator.
 - `dx` could be either `input` or `output`.


### dots and bars: ``\dot{y} = \dfrac{∂y}{∂x} = \overline{x}``
 - `v̇` is a derivative of the input moving forward: ``v̇ = \frac{∂v}{∂x}`` for input ``x``, intermediate value ``v``.
 - `v̄` is a derivative of the output moving backward: ``v̄ = \frac{∂y}{∂v}`` for output ``y``, intermediate value ``v``.

### others
 - `Ω` is often used as the return value of the function. Especially, but not exclusively, for scalar functions.
     - `ΔΩ` is thus a seed for the pullback.
     - `∂Ω` is thus the output of a pushforward.


## Why does `rrule` return the primal function evaluation?
You might wonder why `frule(f, x)` returns `f(x)` and the derivative of `f` at `x`, and similarly for `rrule` returning `f(x)` and the pullback for `f` at `x`.
Why not just return the pushforward/pullback, and let the user call `f(x)` to get the answer separately?

There are three reasons the rules also calculate the `f(x)`.
1. For some rules an alternative way of calculating `f(x)` can give the same answer while also generating intermediate values that can be used in the calculations required to propagate the derivative.
2. For many `rrule`s the output value is used in the definition of the pullback. For example `tan`, `sigmoid` etc.
3. For some `frule`s there exists a single, non-separable operation that will compute both derivative and primal result. For example, this is the case for many of the methods for [differential equation sensitivity analysis](https://docs.juliadiffeq.org/stable/analysis/sensitivity/#sensitivity-1).

For more information and examples see the [design notes on changing the primal](@ref change_primal).

## Where are the derivatives for keyword arguments?

_Pullbacks_ do not return a sensitivity for keyword arguments;
similarly, _pushforwards_ do not accept a perturbation for keyword arguments.
This is because in practice functions are very rarely differentiable with respect to keyword arguments.

As a rule, keyword arguments tend to control side-effects, like logging verbosity,
or to be functionality-changing to perform a different operation, e.g. `dims=3`, and thus not differentiable.

To the best of our knowledge no Julia AD system, with support for the definition of custom primitives, supports differentiating with respect to keyword arguments.
At some point in the future ChainRules may support these. Maybe.


## [What is the difference between `ZeroTangent` and `NoTangent` ?](@id faq_abstract_zero)
`ZeroTangent` and `NoTangent` act almost exactly the same in practice: they result in no change whenever added to anything.
Odds are if you write a rule that returns the wrong one everything will just work fine.
We provide both to allow for clearer writing of rules, and easier debugging.

`ZeroTangent()` represents the fact that if one perturbs (adds a small change to) the matching primal, there will be no change in the behaviour of the primal function.
For example, in `fst(x, y) = x`, the derivative of `fst` with respect to `y` is `ZeroTangent()`.
`fst(10, 5) == 10` and if we add `0.1` to `5` we still get `fst(10, 5.1) == 10`.

`NoTangent()` represents the fact that if one perturbs the matching primal, the primal function will now error.
For example, in `access(xs, n) = xs[n]`, the derivative of `access` with respect to `n` is `NoTangent()`.
`access([10, 20, 30], 2) == 20`, but if we add `0.1` to `2` we get `access([10, 20, 30], 2.1)` which errors as indexing can't be applied at fractional indexes.

## Why do I get an error involving `nothing`?

When no custom `frule` or `rrule` exists, if you try to call one of those, it will return `nothing` by default.
As a result, you may encounter errors like

```julia
MethodError: no method matching iterate(::Nothing)
```

Sometimes you think you have implemented the right rule, but it is called with a slightly different set of arguments than you expected.
You can use [Cthulhu.jl](https://github.com/JuliaDebug/Cthulhu.jl) to dive into the call stack and figure out which method you are missing.

An alternative is to call back into AD: read the documentation on [rule configuration](@ref config) to know more.

## When to use ChainRules vs ChainRulesCore?

[ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) is a light-weight dependency for defining rules for functions in your packages, without you needing to depend on ChainRules.jl itself.
It has almost no dependencies of its own.
If you only want to define rules, not use them, then you probably only want to load ChainRulesCore.jl.

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides the full functionality for AD systems. In particular, it has all the rules for Base Julia and the standard libraries.
It is thus a much heavier package to load.
AD systems making use of `frule`s and `rrule`s should load ChainRules.jl.

## Where should I put my rules?

We recommend adding custom rules to your own packages with [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
It is good to have them in the same package that defines the original function.
This avoids type-piracy, and makes it easy to keep in-sync.
ChainRulesCore is a very light-weight dependency.

## How do I test my rules?

You can use [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl) to test your custom rules.
ChainRulesTestUtils.jl has some dependencies, so it is a separate package from ChainRulesCore.jl.
This means your package can depend on the light-weight ChainRulesCore.jl, and make ChainRulesTestUtils.jl a test-only dependency.

Remember to read the section [On writing good `rrule` / `frule` methods](@ref).

## Is removing a thunk a breaking change?
Removing thunks is not considered a breaking change.
This is because (in principle) removing them changes the implementation of the values
returned by an `rrule`, not the value that they represent.
This is morally the same as similar issues [discussed in ColPrac](https://github.com/SciML/ColPrac#changes-that-are-not-considered-breaking), such as details of floating point arithmetic changing.

On a practical level, it's important that this is the case because thunks are a bit of a hack,
and over time it is hoped that the need for them will reduce, as they increase
code-complexity and place additional stress on the compiler.

## Where can I learn more about AD ?
There are not so many truly excellent learning resources for autodiff out there in the world, which is a bit sad.
The list here is incomplete, but is vetted for quality.

 - [Automatic Differentiation for Dummies keynote video](https://www.youtube.com/watch?v=FtnkqIsfNQc) by [Simon Peyton Jones](https://github.com/simonpj): particularly good if you like pure math type thinking.

 - ["What types work with differentiation?](https://github.com/google-research/dex-lang/issues/454#issuecomment-766089519) comment on DexLang GitHub issue by [Dan Zheng](https://github.com/dan-zheng): summarizes several years of insights from the Swift AD work.

 - MIT 18337 lecture notes 8-10 by [Christopher Rackauckas](https://github.com/ChrisRackauckas) and [David P. Sanders](https://github.com/dpsanders): moves fast from basic to advanced, particularly good if you like applicable mathematics
   - [Automatic Differentiation and Application](https://mitmath.github.io/18337/lecture8/automatic_differentiation): Good introduction
   - [Solving Stiff Ordinary Differential Equations](https://mitmath.github.io/18337/lecture9/stiff_odes): ignore the ODE stuff, most of this is about Sparse AutoDiff, can skip/skim this one
   - [Basic Parameter Estimation, Reverse-Mode AD, and Inverse Problems](https://mitmath.github.io/18337/lecture10/estimation_identification): use in optimization, and details connections of other math.
   - [Differentiable Programming and Neural Differential Equations](https://mitmath.github.io/18337/lecture11/adjoints): Includes custom primitive derivations for equation solvers.

 - [Diff-Zoo Jupyter Notebook Book](https://github.com/MikeInnes/diff-zoo) by [Mike Innes](https://github.com/MikeInnes/diff-zoo), has implementations and explanations.

 - ["Evaluating Derivatives"](https://dl.acm.org/doi/book/10.5555/1455489) by Griewank and Walther is the best book at least for reverse-mode. It also covers forward-mode though (by its own admission) not as well, it never mentioned dual numbers which is an unfortunate lack.
