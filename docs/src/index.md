# ChainRules

[Automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) is a set of techniques for obtaining derivatives of arbitrary functions.
There are surprisingly many packages for doing AD in Julia.
ChainRules isn't one of these packages.

The AD packages essentially combine derivatives of simple functions into derivatives of more complicated functions.
They differ in the way they break down complicated functions into simple ones, but they all require a common set of derivatives of simple functions (rules).

[ChainRules](https://github.com/JuliaDiff/ChainRules.jl) is an AD-independent set of rules, and a system for defining and testing rules.

!!! note "What is a rule?"
    A rule encodes knowledge about propagating derivatives, e.g. that the derivative (with respect to `x`) of `a*x` is `a`, and the derivative of `sin(x)` is `cos(x)`, etc.

## ChainRules ecosystem organisation

The ChainRules ecosystem comprises:
- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl): a system for defining rules, and a collection of tangent types.
- [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl): a collection of rules for Julia Base and standard libraries.
- [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl): utilities for testing rules using finite differences.

AD systems depend on ChainRulesCore.jl to get access to tangent types and the core rule definition functionality (`frule` and `rrule`), and on ChainRules.jl to benefit from the collection of rules for Julia Base and the standard libraries.

Packages that just want to define rules only need to depend on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), which is an exceptionally light dependency.
They should also have a test-only dependency on [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl) to test the rules using finite differences.

Note that the packages with rules do not have to depend on AD systems, and neither do the AD systems have to depend on individual packages.

## Key functionality

Consider a relationship $y = f(x)$, where $f$ is some function.
Computing $y$ from $x$ is the original problem, called the _primal_ computation, in contrast to the problem of computing derivatives.
We say that the _primal function_ $f$ takes a _primal input_ $x$ and returns the _primal output_ $y$.

ChainRules rules are concerned with propagating _tangents_ of primal inputs to _tangents_ of primal outputs (`frule`, from forwards mode AD), and propagating _cotangents_ of primal outputs to _cotangents_ of primal inputs (`rrule`, from reverse mode AD).
To be able to do that, ChainRules also defines a small number of tangent types to represent tangents and cotangents.

!!! note "Tangents and cotangents"
    Strictly speaking tangents, $ẋ = \frac{dx}{da}$, are propagated in `frule`s, and cotangents, $x̄ = \frac{da}{dx}$, are propagated in `rrule`s.
    However, in practice there is rarely a need to distinguish between the two: both are represented by the same tangent types.
    Thus, except when the detail might clarify, we refer to both as tangents.

!!! terminology "`frule` and `rrule`"
    `frule` and `rrule` are ChainRules specific terms.
    Their exact functioning is fairly ChainRules specific, though other tools have similar functions.
    The core notion is sometimes called _custom AD primitives_, _custom adjoints_, _custom gradients_, _custom sensitivities_.
    The whole field is a mess for terminology.


### Forward-mode AD rules (`frule`s)

If we know the value of $ẋ = \frac{dx}{da}$ for some $a$ and we want to know $ẏ = \frac{dy}{da}$, the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) tells us that $ẏ = \frac{dy}{dx} ẋ$.
Intuitively, we are pushing the derivative forward.
This is the basis for forward-mode AD.

!!! note "frule"
    The `frule` for $f$ encodes how to propagate the tangent of the primal input ($ẋ$) to the tangent of the primal output ($ẏ$).

The `frule` signature for a function `foo(args...; kwargs...)` is
```julia
function frule((Δself, Δargs...), ::typeof(foo), args...; kwargs...)
    ...
    return y, ∂Y
end
```
where `y = foo(args; kwargs...)` is the primal output, and `∂Y` is the result of propagating the input tangents `Δself`, `Δargs...` forwards at the point in the domain of `foo` described by `args`.
This propagation is call the pushforward.
Often we will think of the `frule` as having the primal computation `y = foo(args...; kwargs...)`, and the pushforward `∂Y = pushforward(Δself, Δargs...)`,
even though they are not present in seperate forms in the code.

For example, the `frule` for `sin(x)` is:
```julia
function frule((_, Δx), ::typeof(sin), x)
    return sin(x), cos(x) * Δx
end
```

### Reverse-mode AD rules (`rrule`s)

If we know the value of $ȳ = \frac{da}{dy}$ for some $a$ and we want to know $x̄ = \frac{da}{dx}$, the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) tells us that $x̄ =ȳ \frac{dy}{dx}$.
Intuitively, we are pushing the derivative backward.
This is the basis for reverse-mode AD.

!!! note "rrule"
    The `rrule` for $f$ encodes how to propagate the cotangents of the primal output ($ȳ$) to the cotangent of the primal input ($x̄$).

The `rrule` signature for a function `foo(args...; kwargs...)` is
```julia
function rrule(::typeof(foo), args...; kwargs...)
    ...
    return y, pullback
end
```
where `y` (the primal output) must be equal to `foo(args...; kwargs...)`.
`pullback` is a function to propagate the derivative information backwards at the point in the domain of `foo ` described by `args`.
That pullback function is used like:
`∂self, ∂args... = pullback(Δy)`
Almost always the _pullback_ will be declared locally within the `rrule`, and will be a _closure_ over some of the other arguments, and potentially over the primal result too.

For example, the `rrule` for `sin(x)` is:
```julia
function rrule(::typeof(sin), x)
    sin_pullback(Δy) = (NoTangent(), cos(x)' * Δy)
    return sin(x), sin_pullback
end
```

!!! note "Why `rrule` returns a pullback but `frule` doesn't return a pushforward"
    While `rrule` takes only the arguments to the original function (the primal arguments) and returns a function (the pullback) that operates with the derivative information, the `frule` does it all at once.
    This is because the `frule` fuses the primal computation and the pushforward.
    This is an optimization that allows `frule`s to contain single large operations that perform both the primal computation and the pushforward at the same time (for example solving an ODE).
    This operation is only possible in forward mode (where `frule` is used) because the derivative information needed by the pushforward available with the `frule` is invoked -- it is about the primal function's inputs.
    In contrast, in reverse mode the derivative information needed by the pullback is about the primal function's output.
    Thus the reverse mode returns the pullback function which the caller (usually an AD system) keeps hold of until derivative information about the output is available.

### Tangent types

The types of (co)-tangents depend on the types of the primals.
Scalar primals are represented by scalar tangents (e.g. `Float64` tangent for a `Float64` primal).
Vector, matrix, and higher rank tensor primals can be represented by vector, matrix and tensor tangents.

ChainRules defines a [`Tangent`](@ref) tangent type to represent tangents of `struct`s, `Tuple`s, `NamedTuple`s, and `Dict`s.

Additionally, for signalling semantics, we distinguish between two tangent types representing a zero tangent.
[`NoTangent`](@ref) type represent situtations in which the tangent space does not exist, e.g. an index into an array can not be perturbed.
[`ZeroTangent`](@ref) is used for cases where the tangent happens to be zero, e.g. because the primal argument is not used in the computation.

We also define [`Thunk`](@ref)s to allow certain optimisation.
`Thunk`s are a wrapper over a computation that can potentially be avoided, depending on the downstream use.

See the section on [tangent types](@ref tangents) for more details.

## Example of using ChainRules directly

While ChainRules is largely intended as a backend for autodiff systems, it can be used directly.
In fact, this can be very useful if you can constrain the code you need to differentiate to only use things that have rules defined for.
This was once how all neural network code worked.

Using ChainRules directly also helps get a feel for it.

```jldoctest index; output=false
using ChainRulesCore

function foo(x)
    a = sin(x)
    b = 0.2 + a
    c = asin(b)
    return c
end

# Define rules (alternatively get them for free via `using ChainRules`)
@scalar_rule(sin(x), cos(x))
@scalar_rule(+(x, y), (1.0, 1.0))
@scalar_rule(asin(x), inv(sqrt(1 - x^2)))
# output

```
```jldoctest index
#### Find dfoo/dx via rrules
#### First the forward pass, gathering up the pullbacks
x = 3;
a, a_pullback = rrule(sin, x);
b, b_pullback = rrule(+, 0.2, a);
c, c_pullback = rrule(asin, b)

#### Then the backward pass calculating gradients
c̄ = 1;                    # ∂c/∂c
_, b̄ = c_pullback(c̄);     # ∂c/∂b = ∂c/∂b ⋅ ∂c/∂c
_, _, ā = b_pullback(b̄);  # ∂c/∂a = ∂c/∂b ⋅ ∂b/∂a
_, x̄ = a_pullback(ā);     # ∂c/∂x = ∂c/∂a ⋅ ∂a/∂x
x̄                         # ∂c/∂x = ∂foo/∂x
# output
-1.0531613736418153
```
```jldoctest index
#### Find dfoo/dx via frules
x = 3;
ẋ = 1;              # ∂x/∂x
nofields = ZeroTangent();  # ∂self/∂self

a, ȧ = frule((nofields, ẋ), sin, x);                    # ∂a/∂x = ∂a/∂x ⋅ ∂x/∂x 
b, ḃ = frule((nofields, ZeroTangent(), ȧ), +, 0.2, a);  # ∂b/∂x = ∂b/∂a ⋅ ∂a/∂x
c, ċ = frule((nofields, ḃ), asin, b);                   # ∂c/∂x = ∂c/∂b ⋅ ∂b/∂x
ċ                                                       # ∂c/∂x = ∂foo/∂x
# output
-1.0531613736418153
```
```julia
#### Find dfoo/dx via FiniteDifferences.jl
using FiniteDifferences
central_fdm(5, 1)(foo, x)
# output
-1.0531613736418257

#### Find dfoo/dx via ForwardDiff.jl
using ForwardDiff
ForwardDiff.derivative(foo, x)
# output
-1.0531613736418153

#### Find dfoo/dx via Zygote.jl
using Zygote
Zygote.gradient(foo, x)
# output
(-1.0531613736418153,)
```

## Videos

For people who learn better by video we have a number of videos of talks we have given about the ChainRules project.
Note however, that the videos are frozen in time reflecting the state of the packages at the time they were recorded.
This documentation is the continously updated canonical source.
However, we have tried to note below each video notes on its correctness.

The talks that follow are in reverse chronological order (i.e. most recent video is first).

### EuroAD 2021: ChainRules.jl: AD system agnostic rules for JuliaLang
Presented by Lyndon White.
[Slides](https://www.slideshare.net/LyndonWhite2/euroad-2021-chainrulesjl)

This is the talk to watch if you want to understand why the ChainRules project exists, what its challenges are, and how those have been overcome.
It is intended less for users of the package, and more for people working in the field of AD more generally.
It does also serve as a nice motivation for those first coming across the package as well though.

```@raw html
<div class="video-container">
<iframe src="https://www.youtube.com/embed/B3bC49OmTdk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

Abstract:
> The ChainRules project is a suite of JuliaLang packages that define custom primitives (i.e. rules) for doing AD in JuliaLang.
> Importantly it is AD system agnostic.
> It has proved successful in this goal.
> At present it works with about half a dozen different JuliaLang AD systems.
> It has been a long journey, but as of August 2021, the core packages have now hit version 1.0.
>
> This talk will go through why this is useful, the particular objectives the project had, and the challenges that had to be solved.
> This talk is not intended as an educational guide for users (For that see our 2021 JuliaCon talk: > Everything you need to know about ChainRules 1.0 (https://live.juliacon.org/talk/LWVB39)).
> Rather this talk is to share the insights we have had, and likely (inadvertently) the mistakes we have made, with the wider autodiff community.
> We believe these insights can be informative and useful to efforts in other languages and ecosystems.


### JuliaCon 2021: Everything you need to know about ChainRules 1.0
Presented by Miha Zgubič.
[Slides](https://github.com/mzgubic/ChainRulesTalk/blob/master/ChainRules.pdf)

If you are just wanting to watch a video to learn all about ChainRules and how to use it, watch this one.

!!! note "Slide on opting out is incorrect"
    Slide 42 is incorrect (`@no_rrule sum_array(A::Diagonal)`), in the ChainRulesCore 1.0 release the following syntax is used: `@opt_out rrule(::typeof(sum_array), A::Diagonal)`. This syntax allows us to include rule config information.

```@raw html
<div class="video-container">
<iframe src="https://www.youtube.com/embed/a8ol-1l84gc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

Abstract:
> ChainRules is an automatic differentiation (AD)-independent ecosystem for forward-, reverse-, and mixed-mode primitives. It comprises ChainRules.jl, a collection of primitives for Julia Base, ChainRulesCore.jl, the utilities for defining custom primitives, and ChainRulesTestUtils.jl, the utilities to test primitives using finite differences. This talk provides brief updates on the ecosystem since last year and focuses on when and how to write and test custom primitives.


### JuliaCon 2020: ChainRules.jl
Presented by Lyndon White.
[Slides](https://raw.githack.com/oxinabox/ChainRulesJuliaCon2020/main/out/build/index.html)

This talk is primarily of historical interest.
This was the first public presentation of ChainRules.
Though the project was a few years old by this stage.
A lot of things are still the same; conceptually, but a lot has changed.
Most people shouldn't watch this talk now.

!!! warning "Outdated Terminology"
    A lot of terminology has changed since this presentation.
     - `DoesNotExist` → `NoTangent`
     - `Zero` →  `ZeroTangent`
     - `Composite{P}` → `Tangent{T}`
    The talk also says differential in a lot of places where we now would say tangent.

```@raw html
<div class="video-container">
<iframe src="https://www.youtube.com/embed/B4NfkkkJ7rs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

Abstract:
> The ChainRules project allows package authors to write rules for custom sensitivities (sometimes called custom adjoints) in a way that is not dependent on any particular autodiff (AD) package.
> It allows authors of AD packages to access a wealth of prewritten custom sensitivities, saving them the effort of writing them all out themselves.
> ChainRules is the successor to DiffRules.jl and is the native rule system currently used by ForwardDiff2,  Zygote and soon ReverseDiff
