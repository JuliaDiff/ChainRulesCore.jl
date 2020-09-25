# ChainRules Glossary

This glossary serves as a quick reference for common terms used in the field of Automatic Differentiation, as well as those used throughout the documentation relating specifically to ChainRules.

##Definitions:

###Adjoint:

The conjugate transpose of the Jacobian for a given function `f`.

###Derivative:

The derivative of a function `y = f(x)` with respect to the independent variable `x` denoted `f'(x)` or `dy/dx` is the rate of change of the dependent variable `y` with respect to the change of the independent variable `x`. In multiple dimensions, we may refer to the gradient of a function, or its directional derivative.

###Differential:

The differential of a given function `y = f(x)` denoted `dy` is the product of the derivative function `f'(x)` and the increment of the independent variable `dx`. In multiple dimensions, it is the sum of these products across each dimension (using the partial derivative and the given independent variable's increment).

###Directional Derivative:

The directional derivative of a function `f` at any given point in any given unit-direction is the gradient multiplied by the direction. It represents the rate of change of `f` in that direction.

###F-rule:

A function used in forward-mode differentiation. For a given function `f`, it takes in the positional and keyword arguments of `f` and returns the primal result and the pushforward.

###Gradient:

The gradient of a scalar function `f` represented by `∇f` is a vector function whose components are the partial derivatives of `f` with respect to each dimension of the domain of `f`.

###Jacobian:

The Jacobian of a vector-valued function `f` is the matrix of `f`'s first-order partial derivatives.

###Jacobian Transpose Vector Product (j'vp):

The product of the adjoint of the Jacobian and the vector in question. A description of the pullback in terms of its Jacobian.

###Jacobian Vector Product (jvp):

The product of the Jacobian and the vector in question. It is a description of the pushforward in terms of its Jacobian.

###Primal:

Something relating to the original problem, as opposed to relating to the derivative. For example in `y = f(x)`, `f` is the primal function, and computing `f(x)` is doing the primal computation. `y` is the primal return, and `x` is a primal argument. `typeof(y)` and `typeof(x)` are both primal types.

###Pullback:

`Pullback(f)` describes the sensitivity of the input of `f` as a function of (for the relative change to) the sensitivity of the output of `f`. Can be represented as the dot product of a vector (left) the adjoint Jacobian (right).

###Pushforward:

`Pushforward(f)` describes the sensitivity of the output of `f` as a function of (for the relative change to) the sensitivity of the input of `f`. Can be represented as the dot product of the Jacobian (left) and a vector (right).

###R-rule:

A function used in reverse-mode differentiation. For a given function `f`, it takes in the positional and keyword arguments of `f` and returns the primal result and the pullback.
