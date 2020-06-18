# FAQ

## What is up with the different symbols?

### `Δx`, `∂x`, `dx`
ChainRules uses these perhaps atypically.
As a notation that is the same across propagators, regardless of direction (incontrast see `ẋ` and `x̄` below).

 - `Δx` is the input to a propagator, (i.e a _seed_ for a _pullback_; or a _perturbation_ for a _pushforward_)
 - `∂x` is the output of a propagator
 - `dx` could be either `input` or `output`


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
3. For some `frule`s there exists a single, non-separable operation that will compute both derivative and primal result. For example many of the methods for [differential equation sensitivity analysis](https://docs.juliadiffeq.org/stable/analysis/sensitivity/#sensitivity-1).

## Where are the derivatives for keyword arguments?
_pullbacks_ do not return a sensitivity for keyword arguments;
similarly _pushfowards_ do not accept a perturbation for keyword arguments.
This is because in practice functions are very rarely differentiable with respect to keyword arguments.
As a rule keyword arguments tend to control side-effects, like logging verbosity,
or to be functionality changing to perform a different operation, e.g. `dims=3`, and thus not differentiable.
To the best of our knowledge no Julia AD system, with support for the definition of custom primitives, supports differentiating with respect to keyword arguments.
At some point in the future ChainRules may support these. Maybe.


## What is the difference between `Zero` and `DoesNotExist` ?
`Zero` and `DoesNotExist` act almost exactly the same in practice: they result in no change whenever added to anything.
Odds are if you write a rule that returns the wrong one everything will just work fine.
We provide both to allow for clearer writing of rules, and easier debugging.

`Zero()` represents the fact that if one perturbs (adds a small change to) the matching primal there will be no change in the behavour of the primal function.
For example in `fst(x,y) = x`, then the derivative of `fst` with respect to `y` is `Zero()`.
`fst(10, 5) == 10` and if we add `0.1` to `5` we still get `fst(10, 5.1)=10`.

`DoesNotExist()` represents the fact that if one perturbs the matching primal, the primal function will now error.
For example in `access(xs, n) = xs[n]` then the derivative of `access` with respect to `n` is `DoesNotExist()`.
`access([10, 20, 30], 2) = 20`, but if we add `0.1` to `2` we get `access([10, 20, 30], 2.1)` which errors as indexing can't be applied at fractional indexes.


## When to use ChainRules vs ChainRulesCore?

[ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) is a light-weight dependency for defining rules for functions in your packages, without you needing to depend on ChainRules.jl itself.
It has almost no dependencies of its own.
If you only want to define rules, not use them, then you probably only want to load ChainRulesCore.jl.

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides the full functionality for AD systems, in particular it has all the rules for Base Julia and the standard libraries.
It is thus a much heavier package to load.
AD systems making use of `frule`s and `rrule`s should load ChainRules.jl.

## Where should I put my rules?

We recommend adding custom rules to your own packages with [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), rather than adding them to ChainRules.jl.
A few packages - currently SpecialFunctions.jl and NaNMath.jl - have rules in ChainRules.jl as a short-term measure.

## How do I test my rules?

You can use [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl) to test your custom rules.
ChainRulesTestUtils.jl has some dependencies, so it is a separate package from ChainRulesCore.jl.
This means your package can depend on the light-weight ChainRulesCore.jl, and make ChainRulesTestUtils.jl a test-only dependency.

Remember to read the section on [On writing good `rrule` / `frule` methods](@ref).

## How do chain rules work for complex functions?

`ChainRules.jl` follows the convention that `frule` applied to a function ``f(x + i y) = u(x,y) + i v(x,y)`` with perturbation ``\Delta x + i \Delta y`` returns the value and
```math
\tfrac{\partial u}{\partial x} \, \Delta x + \tfrac{\partial u}{\partial y} \, \Delta y + i \, \Bigl( \tfrac{\partial v}{\partial x} \, \Delta x + \tfrac{\partial v}{\partial y} \, \Delta y \Bigr)
.
```
Similarly, `rrule` applied to the same function returns the value and a pullback which, when applied to the adjoint ``\Delta u + i \Delta v``, returns
```math
\Delta u \, \tfrac{\partial u}{\partial x} + \Delta v \, \tfrac{\partial v}{\partial x} + i \, \Bigl(\Delta u \, \tfrac{\partial u }{\partial y} + \Delta v \, \tfrac{\partial v}{\partial y} \Bigr)
.
```
If we interpret complex numbers as vectors in ``\mathbb{R}^2``, then these rules correspond to left- and right-multiplication with the Jacobian of ``f(z)``, i.e. `frule` corresponds to
```math
\begin{pmatrix}
\tfrac{\partial u}{\partial x} \, \Delta x + \tfrac{\partial u}{\partial y} \, \Delta y
\\
\tfrac{\partial v}{\partial x} \, \Delta x + \tfrac{\partial v}{\partial y} \, \Delta y
\end{pmatrix}
=
\begin{pmatrix}
\tfrac{\partial u}{\partial x} & \tfrac{\partial u}{\partial y} \\
\tfrac{\partial v}{\partial x} & \tfrac{\partial v}{\partial y} \\
\end{pmatrix}
\begin{pmatrix}
\Delta x \\ \Delta y
\end{pmatrix}

```
and `rrule` corresponds to
```math
\begin{pmatrix}
\Delta u \, \tfrac{\partial u}{\partial x} + \Delta v \, \tfrac{\partial v}{\partial x}
&
\Delta u \, \tfrac{\partial u}{\partial y} + \Delta v \, \tfrac{\partial v}{\partial y}
\end{pmatrix}
=
\begin{pmatrix}
\Delta u & \Delta v
\end{pmatrix}
\begin{pmatrix}
\tfrac{\partial u}{\partial x} & \tfrac{\partial u}{\partial y} \\
\tfrac{\partial v}{\partial x} & \tfrac{\partial v}{\partial y} \\
\end{pmatrix}
.
```
The Jacobian of ``f:\mathbb{C} \to \mathbb{C}`` interpreted as a function ``\mathbb{R}^2 \to \mathbb{R}^2`` can hence be evaluated using
```
function jacobian_via_frule(f,z)
    fz,df_dx = frule((Zero(), 1),f,z)
    fz,df_dy = frule((Zero(),im),f,z)
    return [
        real(df_dx)  real(df_dy)
        imag(df_dx)  imag(df_dy)
    ]
end
```
```
function jacobian_via_rrule(f,z)
    fz, pullback = rrule(f,z)
    _,du_dz = pullback( 1)
    _,dv_dz = pullback(im)
    return [
        real(du_dz)  imag(du_dz)
        real(dv_dz)  imag(dv_dz)
    ]
end
```

If ``f(z)`` is holomorphic, then the derivative part of `frule` can be implemented as ``f'(z) \, \Delta z`` and the derivative part of `rrule` can be implemented as ``\Delta f \,  \overline{f'(z)}``.
Consequently, holomorphic derivatives can be evaluated using
```
function holomorphic_derivative_via_frule(f,z)
    fz,df_dz = frule((Zero(),1),f,z)
    return df_dz
end
```
```
function holomorphic_derivative_via_rrule(f,z)
    fz, pullback = rrule(f,z)
    dself, conj_df_dz = pullback(1)
    return conj(conj_df_dz)
end
```

!!! note
    There are various notions of complex derivatives (holomorphic and Wirtinger derivatives, Jacobians, gradients, etc.) which often differ in subtle but important ways.
    The goal of ChainRules is to provide the basic differentiation rules upon which these derivatives can be implemented, but it does not implement these derivatives itself.
    It is recommended that you carefully check how the above definitions of `frule` and `rrule` translate into your specific notion of complex derivative since getting this wrong will quietly give you the wrong result.