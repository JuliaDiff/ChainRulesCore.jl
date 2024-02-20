#####
##### `AbstractTangent`
#####

"""
The subtypes of `AbstractTangent` define a custom \"algebra\" for chain
rule evaluation that attempts to factor various features like complex derivative
support, broadcast fusion, zero-elision, etc. into nicely separated parts.

In general a tangent type is the type of a derivative of a value.
The type of the value is for contrast called the primal type.
Differential types correspond to primal types, although the relation is not one-to-one.
Subtypes of  `AbstractTangent` are not the only tangent types.
In fact for the most common primal types, such as `Real` or `AbstractArray{Real}` the
the tangent type is the same as the primal type.

In a circular definition: the most important property of a tangent is that it should
be able to be added (by defining `+`) to another tangent of the same primal type.
That allows for gradients to be accumulated.

It generally also should be able to be added to a primal to give back another primal, as
this facilitates gradient descent.

All subtypes of `AbstractTangent` implement the following operations:

 - `+(a, b)`: linearly combine tangent `a` and tangent `b`
 - `*(a, b)`: multiply the tangent `b` by the scaling factor `a`
 -  `Base.zero(x) = ZeroTangent()`: a zero.

Further, they often implement other linear operators, such as `conj`, `adjoint`, `dot`.
Pullbacks/pushforwards are linear operators, and their inputs are often
`AbstractTangent` subtypes.
Pullbacks/pushforwards in-turn call other linear operators on those inputs.
Thus it is desirable to have all common linear operators work on `AbstractTangent`s.
"""
abstract type AbstractTangent end

Base.:+(x::AbstractTangent) = x

@inline Base.conj(x::AbstractTangent) = x

Base.:/(x::AbstractTangent, y) = x * inv(y)
Base.:\(x, y::AbstractTangent) = inv(x) * y