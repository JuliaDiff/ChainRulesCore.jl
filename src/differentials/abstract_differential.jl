#####
##### `AbstractDifferential`
#####

"""
The subtypes of `AbstractDifferential` define a custom \"algebra\" for chain
rule evaluation that attempts to factor various features like complex derivative
support, broadcast fusion, zero-elision, etc. into nicely separated parts.

In general a differential type is the type of a derivative of a value.
The type of the value is for contrast called the primal type.
Differential types correspond to primal types, although the relation is not one-to-one.
Subtypes of  `AbstractDifferential` are not the only differential types.
In fact for the most common primal types, such as `Real` or `AbstractArray{Real}` the
the differential type is the same as the primal type.

In a circular definition: the most important property of a differential is that it should
be able to be added (by defining `+`) to another differential of the same primal type.
That allows for gradients to be accumulated.

It generally also should be able to be added to a primal to give back another primal, as
this facilitates gradient descent.

All subtypes of `AbstractDifferential` implement the following operations:

 - `+(a, b)`: linearly combine differential `a` and differential `b`
 - `*(a, b)`: multiply the differential `b` by the scaling factor `a`
 -  `Base.zero(x) = Zero()`: a zero.

Further, they often implement other linear operators, such as `conj`, `adjoint`, `dot`.
Pullbacks/pushforwards are linear operators, and their inputs are often
`AbstractDifferential` subtypes.
Pullbacks/pushforwards in-turn call other linear operators on those inputs.
Thus it is desirable to have all common linear operators work on `AbstractDifferential`s.
"""
abstract type AbstractDifferential end

Base.:+(x::AbstractDifferential) = x

"""
    extern(x)

Makes a best effort attempt to convert a differential into a primal value.
This is not always a well-defined operation.
For two reasons:
 - It may not be possible to determine the primal type for a given differential.
 For example, `Zero` is a valid differential for any primal.
 - The primal type might not be a vector space, thus might not be a valid differential type.
 For example, if the primal type is `DateTime`, it's not a valid differential type as two
 `DateTime` can not be added (fun fact: `Milisecond` is a differential for `DateTime`).

Where it is defined the operation of `extern` for a primal type `P` should be
`extern(x) = zero(P) + x`.

!!! note
    Because of its limitations, `extern` should only really be used for testing.
    It can be useful, if you know what you are getting out, as it recursively removes
    thunks, and otherwise makes outputs more consistent with finite differencing.

    The more useful action in general is to call `+`, or in the case of a [`Thunk`](@ref)
    to call [`unthunk`](@ref).

!!! warning
    `extern` may return an alias (not necessarily a copy) to data
    wrapped by `x`, such that mutating `extern(x)` might mutate `x` itself.
"""
@inline extern(x) = x

@inline Base.conj(x::AbstractDifferential) = x
