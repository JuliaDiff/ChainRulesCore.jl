# Mutation Support

ChainRulesCore.jl offers experimental support for mutation, targeting use in forward mode AD.
(Mutation support in reverse mode AD is more complicated and will likely require more changes to the interface)

!!! warning "Experimental"
    This page documents an experimental feature.
    Expect breaking changes in minor versions while this remains.
    It is not suitable for general use unless you are prepared to modify how you are using it each minor release.
    It is thus suggested that if you are using it to use _tilde_ bounds on supported minor versions.


## `MutableTangent`
The [`MutableTangent`](@ref) type is designed to be a partner to the [`Tangent`](@ref) type, with specific support for being mutated in place.
It is required to be a structural tangent, having one tangent for each field of the primal object.

Technically, not all `mutable struct`s need to use `MutableTangent` to represent their tangents.
Just like not all `struct`s need to use `Tangent`s.
Common examples away from this are natural tangent types like for arrays.
However, if one is setting up to use a custom tangent type for this it is sufficiently off the beaten path that we can not provide much guidance.

## `zero_tangent`

The [`zero_tangent`](@ref) function functions to give you a zero (i.e. additive identity) for any primal value.
The [`ZeroTangent`](@ref) type also does this.
The difference is that [`zero_tangent`](@ref) is in general full structural tangent mirroring the structure of the primal.
To be technical the promise of [`zero_tangent`](@ref) is that it will be a value that supports mutation.
However, in practice[^1] this is achieved through in a structural tangent
For mutation support this is important, since it means that there is mutable memory available in the tangent to be mutated when the primal changes.
To support this you thus need to make sure your zeros are created in various places with [`zero_tangent`](@ref) rather than []`ZeroTangent`](@ref).



It is also useful for reasons of type stability, since it forces a consistent type (generally a structural tangent) for any given primal type.
For this reason AD system implementors might chose to use this to create the tangent for all literal values they encounter, mutable or not,
and to process the output of `frule`s to convert [`ZeroTangent`](@ref) into corresponding [`zero_tangent`](@ref)s.

## Writing a frule for a mutating function
It is relatively straight forward to write a frule for a mutating function.
There are a few key points to follow:
 - There must be a mutable tangent input for every mutated primal input
 - When the primal value is changed, the corresponding change must be made to its tangent partner
 - When a value is returned, return its partnered tangent.
 - If (and only if) primal values alias, then their tangents must also alias.

### Example
For example, consider the primal function with:
1. takes two `Ref`s
2. doubles the first one in place
3. overwrites the second one's value with the literal 5.0
4. returns the first one


```julia
function foo!(a::Base.RefValue, b::Base.RefValue)
    a[] *= 2
    b[] = 5.0
    return a
end
```

The frule for this would be:
```julia
function ChainRulesCore.frule((_, ȧ, ḃ), ::typeof(foo!), a::Base.RefValue, b::Base.RefValue)
    @assert ȧ isa MutableTangent{typeof(a)}
    @assert ḃ isa MutableTangent{typeof(b)}

    a[] *= 2
    ȧ.x *= 2  # `.x` is the field that lives behind RefValues

    b[] = 5.0
    ḃ.x = zero_tangent(5.0)  # or since we know that the zero for a Float64 is zero could write `ḃ.x = 0.0`

    return a, ȧ
end
```

Then assuming the AD system does its part to makes sure you are indeed given mutable values to mutate (i.e. those `@assert`ions are true) then all is well and this rule will make mutation correct.

[^1]:
    Further, it is hard to achieve this promise of allowing mutation to be supported without returning a structural tangent.
    Except in the special case of where the struct is not mutable and has no nested fields that are mutable.