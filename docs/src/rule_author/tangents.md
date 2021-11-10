# [Tangent types](@id tangents)

The values that come back from pullbacks or pushforwards are not always the same type as the input/outputs of the primal function.
They are tangents, which correspond roughly to something able to represent the difference between two values of the primal types.
A tangent might be such a regular type, like a `Number`, or a `Matrix`, matching to the original type;
or it might be one of the [`AbstractTangent`](@ref ChainRulesCore.AbstractTangent) subtypes.

Tangents support a number of operations.
Most importantly: `+` and `*`, which let them act as mathematical objects.

The most important `AbstractTangent`s when getting started are the ones about avoiding work:

 - [`Thunk`](@ref): this is a deferred computation. A thunk is a [word for a zero argument closure](https://en.wikipedia.org/wiki/Thunk). A computation wrapped in a `@thunk` doesn't get evaluated until [`unthunk`](@ref) is called on the thunk. `unthunk` is a no-op on non-thunked inputs.
 - [`ZeroTangent`](@ref): It is a special representation of `0`. It does great things around avoiding expanding `Thunks` in addition.

### Other `AbstractTangent`s:
 - [`Tangent{P}`](@ref Tangent): this is the tangent for tuples and  structs. Use it like a `Tuple` or `NamedTuple`. The type parameter `P` is for the primal type.
 - [`NoTangent`](@ref): Zero-like, represents that the operation on this input is not differentiable. Its primal type is normally `Integer` or `Bool`.
 - [`InplaceableThunk`](@ref): it is like a `Thunk` but it can do in-place `add!`.
