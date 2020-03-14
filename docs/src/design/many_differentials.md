# Design Notes: 1-1, vs many-many relationship between differential types and primal types.

ChainRules (and other earlier packages in Julia AD ecosystem) has a system where one primal type (the type being having its derivative taken) can have multiple possible differential types (the type of the derivative); and that one differential type can corresponds to multiple primal types.

This is in-contrast to the Swift AD efforts.
Which has one differential type, per primal type.
(Swift uses the term associated tangent type, rather than differential type).

One thing to understand about differentials is they have to form a vector space (or something very like them).
They need to support addition to each other, they need a zero which doesn't change what it is added to, and they need to support scalar multiplication (this isn't really required, but its handy (e.g. for SGD).
Beyond being a vector space, differentials need to be able to be added to a primal value to get back another primal value.
Or roughly equivalently a differential is a difference between two primal values.

One thing to note in this example is that the *primal* does not have to be a vector space.
We can however always transform it to be a vector space via choosing it an origin and expressing each primal value as a differential typed distance from that origin.

My pet example for thinking about this is to consider the `DateTime`.
A `DateTime` is not a vector space.
It has no zero, they can't be added to each other etc.
The corresponding differential type is any subtype of `Period`.
Such as `Millisecond`, `Hour`, `Day` etc.
We call such differential types the _natural differential_ for that primal.
You will note here that we already have a one primal type to many differential types relationship.
Natural differential types are the types people tend to think in, and thus the type they tend to write custom sensitivity rules in.
A important special case of natural differentials, is that if a primal type is a vector space (e.g. `Real`,`AbstractMatrix`) then it is very common for the natural differential type to be the same type as the primal.
Weirdly though not universally, see `getindex` which is normally a differential `OneHotArray` for all `AbstractArray`s.
(This actually further brings us to weirdness of differential types not actually being directly closed under addition since the sum of `OneHotArray`s is a `SparseArray` or if you add enough, a dense array type).

Now beyond natural differential types, we also have structural differential types.
AutoDiff can not automatically determine natural differential types for a primal.
Though for some things we may be able to declare what they are manually.
Further, some things will not have natural differential types in the first place, e.g. `NamedTuple`, `Tuple`, `WebServer`, `Flux.Dense` etc, so we are destined to make some up.
In contrast to natural differentials, we have  structural differentials.
These are derived from the structure of the input.

ChainRules uses `Composite{P, <:NamedTuple}` to represent a structural differnetial type corresponding to primal type `P`.
`Zygote v0.4` uses `NamedTuple`.
(will be back later to discuss this difference)

So the structural differential is derived from the structure of the input.
Either automatically, as part of the AD, or manually as part of a custom rule

Considering again `DateTime`, lets look at it's structure:
```julia
julia> dump(now())
DateTime
  instant: UTInstant{Millisecond}
    periods: Millisecond
      value: Int64 63719890305605
```
The corresponding structural differential is;
```julia
Composite{DateTime}(
    instant = Composite{UTInstant{Millisecond}}(
        periods = Composite{Millisecond}(
            value = 1 # or whatever value this step is
        )
    )
)
```

So the structural differential is another type of differential.
Since AutoDiff can only create structural differentials (without custom rules);
and since all custom sensitivities are only written in terms of natural differentials  -- since that is what is used in papers about derivatives.
So you need to support both.


Further there are cases, which we can call semi-structural differentials.
Where a there is no natural differential type for the outer most type, but there is for some of its fields.
So the rule author has written a structural differential with some fields that are natural differentials.
Another related case is for types that overload `getproperty` such as `SVD` and `QR`.
In this case the structural differential will be based on the fields, but those fields do not have a easy relation to whats actually used in math, so most rule authors would want to write semi-structural differentials based on the properties.

To return to the question of why ChainRules has `Composite{P, <:NamedTuple}` where as Zygote v0.4 just has `NamedTuple`, it related to this, and being able to overload things more generally.
If one knows that one has a semistructual dervative based on property names: `Composite{QR}(Q=..., R=...)` and one is adding it to true structual deriviative based on field names `Composite{QR}(factors=..., τ=...)`, then we need to overload the addition operator to perform that correctly.
And we can't overload happily similar things for `NamedTuple` since we don't know the type
(infact we can't actually overload addition at all for `NamedTuple` as that would be type-piracy, so have to use `Zygote.accum` instead.)
Another use of the primal being a type parameter is to catch errors. ChainRules disallows the addition of `Composite{SVD}` to `Composite{QR}` since in a correctly differentiated program that can never occur.


There is another kind of unnatural differential.
One that is for computational efficiency.
ChainRules has `Thunk`s and `InplaceThunk`s, which wrap a computation that computes a derivative delaying doing that work until (and if) it is needed (which is indicated via adding it to something or `unthunk`ing manually).
Thus saving time if it is never used.
Another one in this category is `Zero` which represents the hard zero (in Zygote v0.4 this is `nothing`).
For example the derivative of `f(x,y)=2x` with respect to `y` is `Zero()`.
Add `Zero()` to anything, and one gets back the original thing without change.
We noted that all differentials need to be a vector space -- `Zero()` is the [trivial vector space](https://proofwiki.org/wiki/Definition:Trivial_Vector_Space).
Further, add `Zero()` to any primal value (no matter the type) and you get back another value of the smae primal type (the same value infact).
So it meets the requirements of a differential type for *all* primal types.
`Zero` can be a saving on memory since we can avoid allocating anything, and on time since performing the multiplication 
`Zero` and `Thunk` are examples of 1 differential type that is valid for multiple primal types.

Now, you have seen examples of both differential types that work for multiple primal types, and primal types that have  multiple valid differential types.
Semantically we can handle these very easily in julia.
Just put in a few more dispatching on `+`.
Multiple-dispatch is great like that.
The down-side is our type-inference becomes hard.
If you have exactly 1 differential type for each primal type, you can very easily workout what all the types on your reverse pass will be -- you don't really need type inference.
But you lose so so much expressibility.

I don't know how Swift is handling thunks, maybe they are not, maybe they have an optimizing compiler that can just slice out code-paths that don't lead to values that get used; maybe they have a language built in for lazy computation.
They are, as I understand if handling `Zero` by requiring every differential type to define a `zero` method -- which it has since it is a vector space,
which costs memory and time, but probably not actually all that much.

With regards to handling multiple different differential types for one primal, like natural and structural derivatives everything needs to be converted to the canonical differential type of that primal.
As I understand it things can be automatically converted by defining conversion protocols or something like that, so rule authors can return anything that has a conversion protocol to the canonical differential type of the primal.
However, it seems like this will rapidly run into problems.
Recall that the natural differential in the case of `getindex` on an `AbstractArray` was `OneHotArray`.
But for say the standard dense `Array`, the only reasonable canonical differential type is also `Array`.
But if you convert a `OneHotArray` into a dense array you do giant allocations to fill in all the other entries with zero.

So this is the story about why we all many differential types in ChainRules.
