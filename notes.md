# A General Mechanism for Generic Rules for AbstractArrays

That we don't have a general formalism for deriving natural derivatives has been discussed quite a bit recently. As has our lack of understanding of the precise relationship between the generic rrules we're writing, and what AD would do. This PR proposes a recipe for deriving generic rules, which leads to a possible formalism for natural derivatives. This formalism can be applied to any AbstractArray, and AD can be in principle be used to obtain default values for the natural tangent.

I want reviewers to determine whether they agree that the proposed recipe
1. is sufficient for implementing generic rrules on AbstractArrays,
2. is correct, in the sense that it produces the same answer as AD, and 
3. the definition of natural tangents proposed indeed applies to any AbstractArray, and broadly agrees with our intuitions about what a natural tangent should be.

This is a long read. I've tried to condense where possible.



## Starting Point

Imagine a clean-slate version of Zygote / Diffractor, in which any rules writen are either
1. necessary to define what AD should do (e.g. `+` or `*` on `Float64`s, `getindex` on `Array`s, `getfield` on composite types, etc), or
2. produce _exactly_ the same answer that AD would produce -- crucially they always return a structural tangent for a non-primitive composite type.

Moreover, assume that users provide structural tangents -- we'll show how to remove this particular assumption later.

The consequence of the above is that we can safely assume that the input and output of any rule will be either
1. a structural tangent (if the primal is a non-primitive composite type)
2. whatever we have defined its tangent type to be (if it's a primitive type like `Float64` or `Array`).

I'm intentionally not trying to define precisely what a primitive is, but will assume that everyone agrees that `Array`s are an example of a primitive, and that we are happy with representing the tangent of an `Array` by another `Array`.

Assume that structural tangents are a valid representation of a tangent of any non-primitive composite type, convenience for rule-writing aside.

More generally, assume that if Zygote / Diffractor successfully run on a given function under the above assumptions, they give the answer desired (the "correct" answer). Consequently, the core goals of the proposed recipe are to make it possible to both
1. never write a rule which prevents Zygote / Diffractor from differentiating a programme that they already know how to differentiate,
2. make it easy to write rules using intuitive representations of tangents.





## The Formalism

First consider a specific case -- we'll both generalise and optimise the implementation later.

Consider a function `f(x::AbstractArray) -> y::AbstractArray`. Lets assume that there's just one method, so we can be sure that a generic fallback will be hit, regardless the concrete type of the argument.

The intuition behind the recipe is to find a function which is equivalent to `f`, whose rules we know how to write safely. If we can find such a function, AD-ing it will clearly give the correct answer -- the following lays out an approach to doing this.

The recipe is:
1. Map `x` to an `Array`, `x_dense`, using `getindex`. Call this operation `destructure`.
2. Apply `f` to `x_dense` to obtain `y_dense`.
3. Map `y_dense` onto `y`. Call this operation `restructure`.

I'm going to define equivalence of output structurally -- two `AbstractArray`s are equal if
1. they are primitives, and whatever notion of equality we have for them holds, or
2. they are composite types, and each of their fields are equal under this definition.

The reason for this notion of equality is that AD (as proposed above) treats concrete subtypes of AbstractArray no differently from any other composite type.

Step 2 of the recipe is possible to easily write an rrule for, because we know that its arguments are `Array`s. Step 1 is clearly defined for any `AbstractArray`, so we can implement e.g. a pullback for `destructure` which accepts an `Array` and returns a `Tangent`.

Step 3 is the trickier step. We'll get on to it later.

The PR shows how to implement steps 1 and 3 for `Array`s (trivial), `Diagonal`s, and `Symmetric`s.



### The Internals of `f` matter for consistency with AD

Must only use the AbstractArrays API (no access to internal fields, just uses `getindex` and
`size`).
