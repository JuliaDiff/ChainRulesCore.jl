#####
##### `AbstractTangent`
#####

"""
The subtypes of `AbstractTangent` define a custom \"algebra\" for chain
rule evaluation that attempts to factor various features like complex derivative
support, broadcast fusion, zero-elision, etc. into nicely separated parts.

In general a differential type is the type of a derivative of a value.
The type of the value is for contrast called the primal type.
Differential types correspond to primal types, although the relation is not one-to-one.
Subtypes of  `AbstractTangent` are not the only differential types.
In fact for the most common primal types, such as `Real` or `AbstractArray{Real}` the
the differential type is the same as the primal type.

In a circular definition: the most important property of a differential is that it should
be able to be added (by defining `+`) to another differential of the same primal type.
That allows for gradients to be accumulated.

It generally also should be able to be added to a primal to give back another primal, as
this facilitates gradient descent.

All subtypes of `AbstractTangent` implement the following operations:

 - `+(a, b)`: linearly combine differential `a` and differential `b`
 - `*(a, b)`: multiply the differential `b` by the scaling factor `a`
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
