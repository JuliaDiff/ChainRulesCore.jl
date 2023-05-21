# [Design Notes: The many-to-many relationship between tangent types and primal types](@id manytypes)

ChainRules has a system where one primal type (the type having its derivative taken) can have multiple possible tangent types (the type of the derivative); and where one tangent type can correspond to multiple primal types.
This is in-contrast to the Swift AD efforts, which has one tangent type per primal type (Swift uses the term associated tangent type).

!!! terminology "tangent and associated tangent type"
    The use of “associated tangent type” in AD is not technically correct, as they live in the [_cotangent_ plane](https://en.wikipedia.org/wiki/Cotangent_space) instead of the [tangent plane](https://en.wikipedia.org/wiki/Tangent_space).
    However it is often reasonable for AD to treat the cotangent plane and tangent plane as the same thing, and this was an intentional choice by the Swift team.
    In ChainRules we use the term “tangent type” to refer to both tangents and cotangents.

One thing to understand about tangents is that they have to form a [vector space](https://en.wikipedia.org/wiki/Vector_space)  (or something very like them).
They need to support addition to each other, they need a zero which doesn't change what it is added to, and they need to support scalar multiplication (this isn't really required, but it is handy for things like gradient descent).
Beyond being a vector space, tangents need to be able to be added to a primal value to get back another primal value.
Or roughly equivalently a tangent is a difference between two primal values.

One thing to note in this example is that the primal does not have to be a vector.
As an example, consider `DateTime`. A `DateTime` is not a vector space: there is no origin point, and `DateTime`s cannot be added to each other. The corresponding tangent type is any subtype of `Period`, such as `Millisecond`, `Hour`, `Day` etc.
    
## Natural tangent

For a given primal type, we say a natural tangent type is one which people would intuitively think of as representing the difference between two primal values.
It tends to already exist outside of the context of AD.
So `Millisecond`, `Hour`, `Day` etc. are examples of _natural tangents_ for the `DateTime` primal.

Note here that we already have a one primal type to many tangent types relationship.
We have `Millisecond` and `Hour` and `Day` all being valid tangent types for `DateTime`.
In this case we _could_ convert them all to a single tangent type, such as `Nanoseconds`, but that is not always a reasonable decision: we may run in to overflow, or lots of allocations if we need to use a `BigInt` to represent the number of `Nanosecond` since the start of the universe.
For types with more complex semantics, such as array types, these considerations are much more important.

Natural tangent types are the types people tend to think in, and thus the type they tend to write custom sensitivity rules in.
An important special case of natural tangents is when the primal type is a vector space (e.g. `Real`,`AbstractMatrix`) in which case it is _common_ for the natural tangent type to be the same as the primal type.
One exception to this is `getindex`.
The ideal choice of tangent type for `getindex` on a dense array would be some type of sparse array, due to the fact the derivative will have only one non-zero element.
This actually further brings us to a weirdness of tangent types not actually being closed under addition, as it would be ideal for the sparse array to become a dense array if summed over all elements.

## Structural tangent types

AD cannot automatically determine the natural tangent types for a primal. For some types we may be able to declare manually their natural tangent type.
Other types will not have natural tangent types at all - e.g. `NamedTuple`, `Tuple`, `WebServer`, `Flux.Dense` -  so we are destined to make some up.
So beyond _natural_ tangent types, we also have _structural_ tangent types.
ChainRules uses [`Tangent{P, <:NamedTuple}`](@ref Tangent) to represent a structural tangent type corresponding to primal type `P`.
[Zygote](https://github.com/FluxML/Zygote.jl/) v0.4 uses `NamedTuple`.

Structural tangents are derived from the structure of the input.
Either automatically, as part of the AD, or manually, as part of a custom rule.

Consider the structure of `DateTime`:
```julia
julia> dump(now())
DateTime
  instant: UTInstant{Millisecond}
    periods: Millisecond
      value: Int64 63719890305605
```

The corresponding structural tangent is:
```julia
Tangent{DateTime}(
    instant::Tangent{UTInstant{Millisecond}}(
        periods::Tangent{Millisecond}(
            value::Int64
        )
    )
)
```

!!! note "One must be allowed to take derivatives of integer arguments"
    This brings up another contrast to Swift.
    In Swift `Int` is considered non-differentiable, which is quite reasonable; it doesn’t have a very good definition of the limit of a small step (as that would be some floating/fixed point type).
    `Int` is intrinsically discrete.
    It is commonly used for indexing, and if one takes a gradient step, say turning `x[2]` into `x[2.1]` then that is an error.
    However, disallowing `Int` to be used as a tangent means we cannot handle cases like `DateTime` having an inner field of milliseconds counted as an integer from the unix epoch or other cases where an integer is used as a convenience for computational efficiency.
    In the case where a custom sensitivity rule claims that there is a non-zero derivative for an `Int` argument that is being used for indexing, that code is simply wrong.
    We can’t handle incorrect code and trying to is a path toward madness.
    Julia, unlike Swift, is not well suited to handling rules about what you can and can’t do with particular types.

So the structural tangent is another type of tangent.
We must support both natural and structural tangents because AD can only create structural tangents (unless using custom sensitivity rules) and all custom sensitivities are only written in terms of natural tangents, as that is what is used in papers about derivatives.

## Semi-structural tangents

Where there is no natural tangent type for the outermost type but there is for some of its fields, we call this a "semi-structural" tangent.

Consider if we had a representation of a country's GDP as output by some continuous time model like a Gaussian Process, where that representation is as a sequence of `TimeSample`s
structured as follows:
```julia
julia> struct TimeSample
           time::DateTime
           value::Float64
       end
```

We can look at its structure:
```julia
julia> dump(TimeSample(now(), 2.6e9))
TimeSample
  time: DateTime
    instant: Dates.UTInstant{Millisecond}
      periods: Millisecond
        value: Int64 63720043490844
  value: Float64 2.6e9
```

Thus we see the that structural tangent would be:
```julia
Tangent{TimeSample}(
    time::Tangent{DateTime}(
        instant::Tangent{UTInstant{Millisecond}}(
            periods::Tangent{Millisecond}(
                value::Int64
            )
        )
    ),
    value::Float64
)
```

But instead in the custom sensitivity rule we would write a semi-structured tangent type.
Since there is not a natural tangent type for `TimeSample` but there is for `DateTime`.
```julia
Tangent{TimeSample}(
    time::Day,
    value::Float64
)
```

So the rule author has written a structural tangent with some fields that are natural tangents.

Another related case is for types that overload `getproperty` such as `SVD` and `QR`.
In this case the structural tangent will be based on the fields, but those fields do not always have an easy relation to what is actually used in math.
For example, the `QR` type has fields `factors` and `t`, but we would more naturally think in terms of the properties `Q` and `R`.
So most rule authors would want to write semi-structural tangents based on the properties.

To return to the question of why ChainRules has `Tangent{P, <:NamedTuple}` whereas Zygote v0.4 just has `NamedTuple`, it relates to semi-structural derivatives, and being able to overload things more generally.
If one knows that one has a semi-structural derivative based on property names, like `Tangent{QR}(Q=..., R=...)`, and one is adding it to the true structural derivative based on field names `Tangent{QR}(factors=..., τ=...)`, then we need to overload the addition operator to perform that correctly.
We cannot happily overload similar things for `NamedTuple` since we don't know the primal type, only the names of the values contained.
In fact we can't actually overload addition at all for `NamedTuple` as that would be type-piracy, so have to use `Zygote.accum` instead.

Another use of the primal being a type parameter is to catch errors.
ChainRules disallows the addition of `Tangent{SVD}` to `Tangent{QR}` since in a correctly differentiated program that can never occur.

## Tangent types for computational efficiency

There is another kind of unnatural tangent.
One that is for computational efficiency.
ChainRules has [`Thunk`](@ref)s and [`InplaceableThunk`](@ref)s, which wrap the computation of a derivative and delays that work until it is needed, either via the derivative being added to something or being [`unthunk`](@ref)ed manually,
thus saving time if it is never used.

Another tangent type used for efficiency is [`ZeroTangent`](@ref) which represents the hard zero (in Zygote v0.4 this is `nothing`).
For example the derivative of `f(x, y)=2x` with respect to `y` is `ZeroTangent()`.
Add `ZeroTangent()` to anything, and one gets back the original thing without change.
We noted that all tangents need to be a vector space.
 `ZeroTangent()` is the [trivial vector space](https://proofwiki.org/wiki/Definition:Trivial_Vector_Space).
Further, add `ZeroTangent()` to any primal value (no matter the type) and you get back another value of the same primal type (the same value in fact).
So it meets the requirements of a tangent type for *all* primal types.
`ZeroTangent` can save on memory (since we can avoid allocating anything) and on time (since performing the multiplication
`ZeroTangent` and `Thunk` are both examples of a tangent type that is valid for multiple primal types.

## Conclusion

Now, you have seen examples of both tangent types that work for multiple primal types, and primal types that have  multiple valid tangent types.
Semantically we can handle these very easily in julia.
Just put in a few more dispatching on `+`.
Multiple-dispatch is great like that.
The down-side is our type-inference becomes hard.
If you have exactly 1 tangent type for each primal type, you can very easily work out what all the types on your reverse pass will be - you don't really need type inference - but you lose so much expressiveness.

## Appendix: What Swift does

I don't know how Swift is handling thunks, maybe they are not, maybe they have an optimizing compiler that can just slice out code-paths that don't lead to values that get used; maybe they have a language built in for lazy computation.

They are, as I understand it, handling `ZeroTangent` by requiring every tangent type to define a `zero` method -- which it has since it is a vector space.
This costs memory and time, but probably not actually all that much.
With regards to handling multiple different tangent types for one primal, like natural and structural derivatives, everything needs to be converted to the _canonical_ tangent type of that primal.

As I understand it, things can be automatically converted by defining conversion protocols or something like that, so rule authors can return anything that has a conversion protocol to the canonical tangent type of the primal.

However, it seems like this will run into problems.
Recall that the natural tangent in the case of `getindex` on an `AbstractArray` was a sparse array.
But for say the standard dense `Array`, the only reasonable canonical tangent type is also a dense `Array`.
But if you convert a sparse array into a dense array you do giant allocations to fill in all the other entries with zero.

So this is the story about why we have many-to-many tangent types in ChainRules.
