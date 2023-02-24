# [Tangent types](@id tangents)

The values that come back from pullbacks or pushforwards are not always the same type as the input/outputs of the primal function.
They are tangents, which correspond roughly to something able to represent the difference between two values of the primal types.
A tangent might be such a regular type, like a `Number`, or a `Matrix`, matching to the original type;
or it might be one of the [`AbstractTangent`](@ref ChainRulesCore.AbstractTangent) subtypes.

Tangents support a number of operations.
Most importantly: `+` and `*`, which let them act as mathematical objects.

To be more formal they support operations which let them act as a vector space.

## Operations on a tangent type
Any tangent type must support:
 - `zero` which returns an additive identity for that type (though it can just return `ZeroTangent()` (see below))
 - `+` for addition between two tangents of this primal, returning another tangent of this primal. This allows gradient accumulation.
 - `*` for multiplication (scaling) by a scalar.
- `+` between a tangent and its primal type returning another tangent of the primal type -- differential geometers sometimes call this exponential map.

Further they often support other linear operators for convenience in writing rules. 

## The subtypes of AbstractTangent

Not all tangents need to subtype the AbstractTangent type -- in fact most don't: most are just numbers or arrays -- but ChainRulesCore does provide a number of special tangent types that can be very useful. 

 - [`ZeroTangent`](@ref): It is a special representation of `0`. It does great things around avoiding expanding `Thunks` in addition.
 - [`NoTangent`](@ref): Zero-like, represents that the operation on this input is not differentiable. Its primal type is normally `Integer` or `Bool`.
 - [`Tangent{P}`](@ref Tangent): this is the tangent for tuples and  structs. Use it like a `Tuple` or `NamedTuple`. The type parameter `P` is for the primal type.
 - [`Thunk`](@ref): this is a deferred computation. A thunk is a [word for a zero argument closure](https://en.wikipedia.org/wiki/Thunk). A computation wrapped in a `@thunk` doesn't get evaluated until [`unthunk`](@ref) is called on the thunk. `unthunk` is a no-op on non-thunked inputs.
 - [`InplaceableThunk`](@ref): it is like a `Thunk` but it can do in-place `add!` which allows for avoiding allocation during gradient accumulation.
