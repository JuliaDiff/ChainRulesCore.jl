#####
##### `AbstractDifferential`
#####

"""
The subtypes of `AbstractDifferential` define a custom \"algebra\" for chain
rule evaluation that attempts to factor various features like complex derivative
support, broadcast fusion, zero-elision, etc. into nicely separated parts.

All subtypes of `AbstractDifferential` implement the following operations:

`+(a, b)`: linearly combine differential `a` and differential `b`

`*(a, b)`: multiply the differential `b` by the scalaring factor `b`

`Base.conj(x)`: complex conjugate of the differential `x`

`Base.zero(x) = Zero()`: a zero.

In general a differential is the type of a derivative of a value.
The type of the value is for contrast called the primal type.
Differntial types correspond to primal types, though the relation is not one-to-one.
Not only `AbstractDifferential` are valid differentials.
Infact for the most common primal types, such as `Real` or `AbstractArray{Real}` the
the differential type is the same as the primal type.

In a circular definition: the most important property of a differential is that it should
be able to be added (by defining `+`) to other differential for that same primal.
It generally also should be able to be added to a primal to give back another primal, as
this facilitates gradient descent.
"""
abstract type AbstractDifferential end

Base.:+(x::AbstractDifferential) = x

"""
    extern(x)

Makes a best effort attempt to convert a differential, into a primal value.
This is not always a well-defined operation.
For two reasons:
 - It may not be possible to determine the primal type for a given differential.
 For example `Zero` is a valid differential for any primal.
 - The primal type might not be a vector space, thus might not be a valid differential type.
 For example, if the primal type is `DateTime`, its not a valid differential type as two.
 `DateTime` can not be added (fun fact: `Milisecond` is a differential for `DateTime`).

Where it is defined the operation of `extern` sfor a primal type `P` should be
`extern(x) = zero(P) + x`.

Because of its limitations, extern should only really be used for testing.
It can be useful, if you know what you are getting out, as it recursively removes thunks,
and otherwise makes outputs more consistent with finite differencing.
The more useful action in general to call `+`, or in the case of thunks: [`unthunk`](@ref).

Note that `extern` may return an alias (not necessarily a copy) to data
wrapped by `x`, such that mutating `extern(x)` might mutate `x` itself.
"""
@inline extern(x) = x

@inline Base.conj(x::AbstractDifferential) = x
