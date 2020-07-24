# ChainRules

[ChainRules](https://github.com/JuliaDiff/ChainRules.jl) provides a variety of common utilities that can be used by downstream [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) tools to define and execute forward-, reverse-, and mixed-mode primitives.

## Introduction

ChainRules is all about providing a rich set of rules for differentiation.
When a person learns introductory calculus, they learn that the derivative (with respect to `x`) of `a*x` is `a`, and the derivative of `sin(x)` is `cos(x)`, etc.
And they learn how to combine simple rules, via [the chain rule](https://en.wikipedia.org/wiki/Chain_rule), to differentiate complicated functions.
ChainRules is a programmatic repository of that knowledge, with the generalizations to higher dimensions.

[Autodiff (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) tools roughly work by reducing a problem down to simple parts that they know the rules for, and then combining those rules.
Knowing rules for more complicated functions speeds up the autodiff process as it doesn't have to break things down as much.

**ChainRules is an AD-independent collection of rules to use in a differentiation system.**


!!! note "The whole field is a mess for terminology"
    It isn't just ChainRules, it is everyone.
    Internally ChainRules tries to be consistent.
    Help with that is always welcomed.

!!! terminology "Primal"
    Often we will talk about something as _primal_.
    That means it is related to the original problem, not its derivative.
    For example in `y = foo(x)`, `foo` is the _primal_ function, and computing `foo(x)` is doing the _primal_ computation.
    `y` is the _primal_ return, and `x` is a _primal_ argument.
    `typeof(y)` and `typeof(x)` are both _primal_ types.


## `frule` and `rrule`

!!! terminology "`frule` and `rrule`"
    `frule` and `rrule` are ChainRules specific terms.
    Their exact functioning is fairly ChainRules specific, though other tools have similar functions.
    The core notion is sometimes called _custom AD primitives_, _custom adjoints_, _custom gradients_, _custom sensitivities_.

The rules are encoded as `frule`s and `rrule`s, for use in forward-mode and reverse-mode differentiation respectively.

The `rrule` for some function `foo`, which takes the positional arguments `args` and keyword arguments `kwargs`, is written:

```julia
function rrule(::typeof(foo), args...; kwargs...)
    ...
    return y, pullback
end
```
where `y` (the primal result) must be equal to `foo(args...; kwargs...)`.
`pullback` is a function to propagate the derivative information backwards at that point.
That pullback function is used like:
`∂self, ∂args... = pullback(Δy)`


Almost always the _pullback_ will be declared locally within the `rrule`, and will be a _closure_ over some of the other arguments, and potentially over the primal result too.

The `frule` is written:
```julia
function frule((Δself, Δargs...), ::typeof(foo), args...; kwargs...)
    ...
    return y, ∂Y
end
```
where again `y = foo(args; kwargs...)`,
and `∂Y` is the result of propagating the derivative information forwards at that point.
This propagation is call the pushforward.
Often we will think of the `frule` as having the primal computation `y = foo(args...; kwargs...)`, and the pushforward `∂Y = pushforward(Δself, Δargs...)`,
even though they are not present in seperate forms in the code.


!!! note "Why `rrule` returns a pullback but `frule` doesn't return a pushforward"
    While `rrule` takes only the arguments to the original function (the primal arguments) and returns a function (the pullback) that operates with the derivative information, the `frule` does it all at once.
    This is because the `frule` fuses the primal computation and the pushforward.
    This is an optimization that allows `frule`s to contain single large operations that perform both the primal computation and the pushforward at the same time (for example solving an ODE).
This operation is only possible in forward mode (where `frule` is used) because the derivative information needed by the pushforward available with the `frule` is invoked -- it is about the primal function's inputs.
    In contrast, in reverse mode the derivative information needed by the pullback is about the primal function's output.
    Thus the reverse mode returns the pullback function which the caller (usually an AD system) keeps hold of until derivative information about the output is available.


## The propagators: pushforward and pullback


!!! terminology "pushforward and pullback"

    _Pushforward_ and _pullback_ are fancy words that the autodiff community recently adopted from Differential Geometry.
    The are broadly in agreement with the use of [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)) and [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)) in differential geometry.
    But any geometer will tell you these are the super-boring flat cases. Some will also frown at you.
    They are also sometimes described in terms of the jacobian:
    The _pushforward_ is _jacobian vector product_ (`jvp`), and _pullback_ is _jacobian transpose vector product_ (`j'vp`).
    Other terms that may be used include for _pullback_ the **backpropagator**, and by analogy for _pushforward_ the **forwardpropagator**, thus these are the _propagators_.
    These are also good names because effectively they propagate wiggles and wobbles through them, via the chain rule.
    (the term **backpropagator** may originate with ["Lambda The Ultimate Backpropagator"](http://www-bcl.cs.may.ie/~barak/papers/toplas-reverse.pdf) by Pearlmutter and Siskind, 2008)

### Core Idea

#### Less formally

 - The **pushforward** takes a wiggle in the _input space_, and tells what wobble you would create in the output space, by passing it through the function.
 - The **pullback** takes wobbliness information with respect to the function's output, and tells the equivalent wobbliness with respect to the functions input.

#### More formally
The **pushforward** of ``f`` takes the _sensitivity_ of the input of ``f`` to a quantity, and gives the _sensitivity_ of the output of ``f`` to that quantity
The **pullback** of ``f`` takes the _sensitivity_ of a quantity to the output of ``f``, and gives the _sensitivity_ of that quantity to the input of ``f``.

### Math
This is all a bit simplified by talking in 1D.

#### Lighter Math
For a chain of expressions:
```
a = f(x)
b = g(a)
c = h(b)
```

The pullback of `g`, which incorporates the knowledge of `∂b/∂a`,
applies the chain rule to go from `∂c/∂b` to `∂c/∂a`.

The pushforward of `g`,  which also incorporates the knowledge of `∂b/∂a`,
applies the chain rule to go from `∂a/∂x` to `∂b/∂x`.

#### Geometric interpretation of reverse and forwards mode AD

Let us think of our types geometrically. In other words, elements of a type form a _manifold_.
This document will explain this point of view in some detail.

##### Some terminology/conventions

Let ``p`` be an element of type M, which is defined by some assignment of numbers ``x_1,...,x_m``,
say ``(x_1,...,x_m) = (a_1,...,1_m)``

A _function_ ``f:M \to K`` on ``M`` is (for simplicity) a polynomial ``K[x_1, ... x_m]``

The tangent space ``T_pM`` of ``T`` at point ``p`` is the ``K``-vector space spanned by derivations ``d/dx``.
The tangent space acts linearly on the space of functions. They act as usual on functions. Our starting point is
that we know how to write down ``d/dx(f) = df/dx``.

The collection of tangent spaces ``{T_pM}`` for ``p\in M`` is called the _tangent bundle_ of ``M``.

Let ``df`` denote the first order information of ``f`` at each point. This is called the differential of ``f``.
If the derivatives of ``f`` and ``g`` agree at ``p``, we say that ``df`` and ``dg`` represent the same cotangent at ``p``.
The covectors ``dx_1, ..., dx_m`` form the basis of the cotangent space ``T^*_pM`` at ``p``. Notice that this vector space is
dual to ``T_p``

The collection of cotangent spaces ``{T^*_pM}`` for ``p\in M`` is called the _cotangent bundle_ of ``M``.

##### Push-forwards and pullbacks

Let ``N`` be another type, defined by numbers ``y_1,...,y_n``, and let ``g:M \to N`` be a _map_, that is,
an ``n``-dimensional vector ``(g_1, ..., g_m)`` of functions on ``M``.

We define the _push-forward_ ``g_*:TM \to TN`` between tangent bundles by ``g_*(X)(h) = X(g\circ h)`` for any tangent vector ``X`` and function ``f``.
We have ``g_*(d/dx_i)(y_j) = dg_j/dx_i``, so the push-forward corresponds to the Jacobian, given a chosen basis.

Similarly, the pullback of the differential ``df`` is defined by
``g^*(df) = d(f\circ g)``. So for a coordinate differential ``dy_j``, we have
``g^*(dy_j) = d(g_j)``. Notice that this is a covector, and we could have defined the pullback by its action on vectors by
``g^*(dh)(X) = g_*(X)(dh) = X(g\circ h)`` for any function ``f`` on ``N`` and ``X\in TM``. In particular,
``g^*(dy_j)(d/dx_i) = d(g_j)/dx_i``. If you work out the action in a basis of the cotangent space, you see that it acts
by the adjoint of the Jacobian.

Notice that the pullback of a differential and the pushforward of a vector have a very different meaning, and this should
be reflected on how they are used in code.

The information contained in the push-forward map is exactly _what does my function do to tangent vectors_.
Pullbacks, acting on differentials of functions, act by taking the total derivative of a function.
This works in a coordinate invariant way, and works without the notion of a metric.
_Gradients_ recall are vectors, yet they should contain the same information of the differential ``df``.
Assuming we use the standard euclidean metric, we can identify ``df`` and ``\nabla f`` as vectors.
But pulling back gradients still should not be a thing.

If the goal is to evaluate the gradient of a function ``f=g\circ h:M \to N \to K``, where ``g`` is a map and ``h`` is a function,
we have two obvious options:
First, we may push-forward a basis of ``M`` to ``TK`` which we identify with K itself.
This results in ``m`` scalars, representing components of the gradient.
Step-by-step in coordinates:
1. Compute the push-forward of the basis of ``T_pM``, i.e. just the columns of the Jacobian ``dg_i/dx_j``.
2. Compute the push-forward of the function ``h`` (consider it as a map, K is also a manifold!) to get ``h_*(g_*T_pM) = \sum_j dh/dy_i (dg_i/dx_j)``

Second, we pull back the differential ``dh``:
1. compute ``dh = dh/dy_1,...,dh/dy_n`` in coordinates.
2. pull back by (in coordinates) multiplying with the adjoint of the Jacobian, resulting in ``g_*(dh) = \sum_i(dg_i/dx_j)(dh/dy_i)``.


### The anatomy of pullback and pushforward

For our function `foo(args...; kwargs...) = y`:


```julia
function pullback(Δy)
    ...
    return ∂self, ∂args...
end
```

The input to the pullback is often called the _seed_.
If the function is `y = f(x)` often the pullback will be written `s̄elf, x̄ = pullback(ȳ)`.

!!! note

    The pullback returns one `∂arg` per `arg` to the original function, plus one `∂self` for the fields of the function itself (explained below).

!!! terminology "perturbation, seed, sensitivity"
    Sometimes _perturbation_, _seed_, and even _sensitivity_ will be used interchangeably.
    They are not generally synonymous, and ChainRules shouldn't mix them up.
    One must be careful when reading literature.
    At the end of the day, they are all _wiggles_ or _wobbles_.


The pushforward is a part of the `frule` function.
Considered alone it would look like:

```julia
function pushforward(Δself, Δargs...)
    ...
    return ∂y
end
```
But because it is fused into frule we see it as part of:
```julia
function frule((Δself, Δargs...), ::typeof(foo), args...; kwargs...)
    ...
    return y, ∂y
end
```


The input to the pushforward is often called the _perturbation_.
If the function is `y = f(x)` often the pushforward will be written `ẏ = last(frule((ṡelf, ẋ), f, x))`.
`ẏ` is commonly used to represent the perturbation for `y`.

!!! note

    In the `frule`/pushforward,
    there is one `Δarg` per `arg` to the original function.
    The `Δargs` are similar in type/structure to the corresponding inputs `args` (`Δself` is explained below).
    The `∂y` are similar in type/structure to the original function's output `Y`.
    In particular if that function returned a tuple then `∂y` will be a tuple of the same size.

### Self derivative `Δself`, `∂self`, `s̄elf`, `ṡelf` etc

!!! terminology "Δself, ∂self, s̄elf, ṡelf"
    It is the derivatives with respect to the internal fields of the function.
    To the best of our knowledge there is no standard terminology for this.
    Other good names might be `Δinternal`/`∂internal`.

From the mathematical perspective, one may have been wondering what all this `Δself`, `∂self` is.
Given that a function with two inputs, say `f(a, b)`, only has two partial derivatives:
``\dfrac{∂f}{∂a}``, ``\dfrac{∂f}{∂b}``.
Why then does a `pushforward` take in this extra `Δself`, and why does a `pullback` return this extra `∂self`?

The reason is that in Julia the function `f` may itself have internal fields.
For example a closure has the fields it closes over; a callable object (i.e. a functor) like a `Flux.Dense` has the fields of that object.

**Thus every function is treated as having the extra implicit argument `self`, which captures those fields.**
So every `pushforward` takes in an extra argument, which is ignored unless the original function has fields.
It is common to write `function foo_pushforward(_, Δargs...)` in the case when `foo` does not have fields.
Similarly every `pullback` returns an extra `∂self`, which for things without fields is the constant `NO_FIELDS`, indicating there are no fields within the function itself.


### Pushforward / Pullback summary

- **Pullback**
   - returned by `rrule`
   - takes output space wobbles, gives input space wiggles
   - Argument structure matches structure of primal function output
   - If primal function returns a tuple, then pullback takes in a tuple of differentials.
   - 1 return per original function argument + 1 for the function itself

- **Pushforward:**
    - part of `frule`
    - takes input space wiggles, gives output space wobbles
    - Argument structure matches primal function argument structure, but passed as a tuple at start of `frule`
    - 1 argument per original function argument + 1 for the function itself
    - 1 return per original function return


### Pullback/Pushforward and Directional Derivative/Gradient

The most trivial use of the `pushforward` from within `frule` is to calculate the [directional derivative](https://en.wikipedia.org/wiki/Directional_derivative):

If we would like to know the directional derivative of `f` for an input change of `(1.5, 0.4, -1)`

```julia
direction = (1.5, 0.4, -1) # (ȧ, ḃ, ċ)
y, ẏ = frule((Zero(), direction...), f, a, b, c)
```

On the basis directions one gets the partial derivatives of `y`:
```julia
y, ∂y_∂a = frule((Zero(), 1, 0, 0), f, a, b, c)
y, ∂y_∂b = frule((Zero(), 0, 1, 0), f, a, b, c)
y, ∂y_∂c = frule((Zero(), 0, 0, 1), f, a, b, c)
```

Similarly, the most trivial use of `rrule` and returned `pullback` is to calculate the [gradient](https://en.wikipedia.org/wiki/Gradient):

```julia
y, f_pullback = rrule(f, a, b, c)
∇f = f_pullback(1)  # for appropriate `1`-like seed.
s̄elf, ā, b̄, c̄ = ∇f
```
Then we have that `∇f` is the _gradient_ of `f` at `(a, b, c)`.
And we thus have the partial derivatives ``\overline{\mathrm{self}}, = \dfrac{∂f}{∂\mathrm{self}}``, ``\overline{a} = \dfrac{∂f}{∂a}``, ``\overline{b} = \dfrac{∂f}{∂b}``, ``\overline{c} = \dfrac{∂f}{∂c}``, including the and the self-partial derivative, ``\overline{\mathrm{self}}``.

## Differentials

The values that come back from pullbacks or pushforwards are not always the same type as the input/outputs of the primal function.
They are differentials, which correspond roughly to something able to represent the difference between two values of the primal types.
A differential might be such a regular type, like a `Number`, or a `Matrix`, matching to the original type;
or it might be one of the [`AbstractDifferential`](@ref ChainRulesCore.AbstractDifferential) subtypes.

Differentials support a number of operations.
Most importantly: `+` and `*`, which let them act as mathematical objects.

The most important `AbstractDifferential`s when getting started are the ones about avoiding work:

 - [`Thunk`](@ref): this is a deferred computation. A thunk is a [word for a zero argument closure](https://en.wikipedia.org/wiki/Thunk). A computation wrapped in a `@thunk` doesn't get evaluated until [`unthunk`](@ref) is called on the thunk. `unthunk` is a no-op on non-thunked inputs.
 - [`One`](@ref), [`Zero`](@ref): There are special representations of `1` and `0`. They do great things around avoiding expanding `Thunks` in multiplication and (for `Zero`) addition.

### Other `AbstractDifferential`s:
 - [`Composite{P}`](@ref Composite): this is the differential for tuples and  structs. Use it like a `Tuple` or `NamedTuple`. The type parameter `P` is for the primal type.
 - [`DoesNotExist`](@ref): Zero-like, represents that the operation on this input is not differentiable. Its primal type is normally `Integer` or `Bool`.
 - [`InplaceableThunk`](@ref): it is like a `Thunk` but it can do in-place `add!`.

 -------------------------------

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
@scalar_rule(+(x, y), (One(), One()))
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
nofields = Zero();  # ∂self/∂self

a, ȧ = frule((nofields, ẋ), sin, x);             # ∂a/∂x = ∂a/∂x ⋅ ∂x/∂x 
b, ḃ = frule((nofields, Zero(), ȧ), +, 0.2, a);  # ∂b/∂x = ∂b/∂a ⋅ ∂a/∂x
c, ċ = frule((nofields, ḃ), asin, b);            # ∂c/∂x = ∂c/∂b ⋅ ∂b/∂x
ċ                                                # ∂c/∂x = ∂foo/∂x
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
