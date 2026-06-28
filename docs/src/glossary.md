# ChainRules Glossary

This glossary serves as a quick reference for common terms used in the field of [Automatic Differentiation](#automatic-differentiation), as well as those used throughout the documentation relating specifically to [`ChainRules.jl`](https://www.juliadiff.org/ChainRulesCore.jl/stable/index.html).

##Definitions:

###Adjoint:

The adjoint is conjugate transpose of the [Jacobian](#jacobian) for a given function `f`.

###Automatic Differentiation:

[Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is the process of applying numerical methods to solving derivative problems, most often algorithmically, using computer programs to achieve high degrees of accuracy.

###Derivative:

The [derivative](https://en.wikipedia.org/wiki/Derivative) of a function `y = f(x)` with respect to the independent variable `x` denoted `f'(x)` or `dy/dx` is the rate of change of the dependent variable `y` with respect to the change of the independent variable `x`. In multiple dimensions, the derivative is not defined.
Instead there are the [partial derivatives](https://en.wikipedia.org/wiki/Partial_derivative), the [directional derivative](#directional-derivative) and the [jacobian](#jacobian) (called [gradient](#gradient) for scalar-valued functions).

###Differential:

The [differential](https://en.wikipedia.org/wiki/Differential_(mathematics)) of a given function `y = f(x)` denoted `dy` is the product of the derivative function `f'(x)` and the increment of the independent variable `dx`. In multiple dimensions, it is the sum of these products across each dimension (using the partial derivative and the given independent variable's increment).

In ChainRules, differentials are types ("differential types") and correspond to primal types. A differential type should represent a difference between two primal typed values.

####Natural Differential:

A natural differential type for a given primal type is a `ChainRules.jl` specific term for the type people would intuitively associate with representing the difference between two values of the primal type. This is in contrast to the [structural differential](#structural-differential).
* **Note:** Not to be confused with the [natural gradient](https://towardsdatascience.com/natural-gradient-ce454b3dcdfa), which is an unrelated concept.

**eg.** A natural differential type for the primal type `DateTime` could be `Hours`

####Structural Differential:

If a given [primal](#primal) type `P` does not have a [natural differential](#natural-differential), we need to come up with one that makes sense. These are called structural differentials and are `ChainRules.jl` specific terms represented as `Composite{P}` and mirrors the structure of the primal type.

####Thunk:

If we wish to delay the computation of a derivative for whatever reason, we wrap it in a [`Thunk`](https://en.wikipedia.org/wiki/Thunk) or `InplaceableThunk`. It holds off on computing the wrapped [derivative](#derivative) until it is needed.

For the purposes of `ChainRles.jl`, the `AbstractThunk` subtype is an "unnatural" differential type. It is a function set up to act like a differential.

####Zero:

The additive [identity](https://en.wikipedia.org/wiki/Identity_(mathematics)) for differentials. It represents the hard zero (ie adding it to anything returns the original thing). `Zero()` can also be a differential type.

###Directional Derivative:

The [directional derivative](https://en.wikipedia.org/wiki/Directional_derivative) of a function `f` at any given point in any given unit-direction is the [gradient](#gradient) multiplied by the direction (ie. the [Jacobian Vector Product](#pushforward)). It represents the rate of change of `f` in the given direction. This gets computed by the [pushforward](#pushforward) function.

###`frule`:

A forward mode rule, that descripes how to propagate the sensitivity into the forwards direction.

The `frule` fuses the [primal](#primal) computation and the pushforward. It takes in the primal function name, the primal arguments and their matching [partial derivatives](https://en.wikipedia.org/wiki/Partial_derivative). It returns the primal output, and the matching [directional derivative](#directional-derivative) (jvp).

###Gradient:

The gradient of a scalar function `f` represented by `∇f` is a vector function whose components are the partial derivatives of `f` with respect to each dimension of the domain of `f`. This is equivalent to the jacobian for scalar-valued functions.

###Jacobian:

The [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of a vector-valued function `f` is the matrix of `f`'s first-order [partial derivatives](https://en.wikipedia.org/wiki/Partial_derivative).

###Primal:

Something relating to the original problem, as opposed to relating to the [derivative](#derivative).
Such as:
 - The primal function being the function that is to be differnetiated
 - The primal inputs being the inputs to that function (the point that the derivative is being calculated at)
 - The primal outputs being the result of applying the primal function to the primal inputs
 - The primal pass (also called the forward pass) where the computation is run to get the primal outputs (generally before doing a derivative (i.e. reverse pass) in reverse mode AD).
 - The primal computation which is the part of the code that is run during the primal pass and must at least compute the primal outputs (but may compute other things to use during the derivative pass).
 - The primal types being the types of the primal inputs/outputs

###Pullback:

[`Pullback(f)`](https://en.wikipedia.org/wiki/Pullback) describes the sensitivity of a quantity to the input of `f` as a function of its sensitivity to the output of `f`. Can be represented as the dot product of a vector and the [adjoint](#adjoint) of the [Jacobian](#jacobian).

####Jacobian Transpose Vector Product (j'vp):

The product of the [adjoint](#adjoint) of the [Jacobian](#jacobian) and the vector in question. A description of the pullback in terms of its Jacobian.

###Pushforward:

[`Pushforward(f)`](https://en.wikipedia.org/wiki/Pushforward) describes the sensitivity of a quantity to the output of `f` as a function of its sensitivity to the input of `f`. Can be represented as the dot product of the [Jacobian](#jacobian) and a vector.

####Jacobian Vector Product (jvp):

The product of the [Jacobian](#jacobian) and the vector in question.

* **Note:**
The jvp is a description of the pushforward in terms of its [Jacobian](#jacobian) and is often used interchangeably with the term pushforward as a result. Strictly speaking, the pushforward computes the jvp (ie the jvp is not normally seen as the name of a function).

###`rrule`:

A reverse mode rule, that descripes how to propagate the sensitivity into the reverse direction.

The `rrule` fuses the [primal](#primal) computation and the pullback. It takes in the primal function name and the primal arguments. It returns the primal output and the propogation rule (j'vp).


