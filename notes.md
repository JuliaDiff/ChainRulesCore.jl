# A General Mechanism for Generic Rules for AbstractArrays

That we don't have a general formalism for deriving natural (co)tangents has been discussed quite a bit recently. As has our lack of understanding of the precise relationship between the generic rrules we're writing, and what AD would do. This PR proposes a recipe for deriving generic rules, which leads to a possible formalism for natural derivatives. This formalism can be applied to any AbstractArray, and AD can be in principle be used to obtain default values for the natural tangent. Moreover, there's some utility functionality proposed to make working with this formalism straightforward for rule-writers.


I want reviewers to determine whether they agree that the proposed recipe
1. is sufficient for implementing generic rrules on AbstractArrays,
2. is correct, in the sense that it produces the same answer as AD, and 
3. the definition of natural tangents proposed indeed applies to any AbstractArray, and broadly agrees with our intuitions about what a natural tangent should be.

I think what is proposed should be doable without making breaking changes since it just involves a changing the output types of some rules, which isn't something that we consider breaking provided that they represent the same thing. I'd prefer we worry about this later if we think this is a good idea though.

Apologies in advance for the lenght. I've tried to condense where possible, but there's a decent amount of content to get through.





## Cutting to the Chase

Under the proposed system, rule-implementers would write rules that look like this:
```julia
function rrule(config::RuleConfig, ::typeof(*), A::AbstractMatrix, B::AbstractMatrix)

    # Do the primal computation.
    C = A * B

    # "natural pullback": write intuitive pullback, closing over stuff in the usual manner.
    function natural_pullback_for_mul(C̄_natural)
        Ā_natural = C̄_natural * B'
        B̄_natural = A' * C̄_natural
        return NoTangent(), Ā_natural, B̄_natural
    end

    # Make a call to utility functionality which transforms cotangents of C, A, and B.
    # Rule-writing without this utility has similar requirements to `ProjectTo`.
    return C, wrap_natural_pullback(config, natural_pullback_for_mul, C, A, B)
end
```
I'm proposing to coin the term "natural pullback" for pullbacks written within this system, as they're rules written involving natural (co)tangents.

Authors will have to implement two functions for their `AbstractArray` type `P`:
```julia
pullback_of_destructure(::P)
pullback_of_restructure(::P)
```
which are the pullbacks of two functions, `destructure` and `(::Restructure)`, that we'll define later.

The proposed candidates for natural (co)tangents are obtained as follows:
1. natural tangents are obtained from structural tangents via the pushforward of `destructure`,
2. natural cotangents are obtained from structural cotangents via the pullback of `(::Restructure)`.

Our current `ProjectTo` functionality is roughly the same as the pullback of `destructure`.

In the proposed system, natural (co)tangents remain confined to `rrule`s, and rule authors can choose to work with either natural, structural, or a mixture of (co)tangents.

Other than the headlines at the top, additional benefits of (a correct implementation of) this approach include:
1. no chance of trying to sum tangents failing because one is natural and the other structural,
1. no risk of obstructing AD,
1. rand_tangent can just use structural tangents, simplifying its implementation and improving its reliability,
1. we can probably finally make `to_vec` treat things structurally (although we also need to extend it in other ways), which will also deal with reliability / simplicity of implementation problems,
1. generic constructor for composite types easy to implement,
1. due to the utility functionality, all of the examples that I've encountered so far are very concise.

The potential downside is additional conversions between natural and structural tangents. Most of the time, these are free. When they're not, you ideally want to minimise them. I'm not sure how often this is going to be a problem, but it's something we're essentially ignoring at the minute (as far as I know), so we're probably going to have to incur some additional cost at some point even if we don't go down the route proposed here.





## Starting Point

Imagine a clean-slate version of Zygote / Diffractor, in which any rules writen are either
1. necessary to define what AD should do (e.g. `+` or `*` on `Float64`s, `getindex` on `Array`s, `getfield` on composite types, etc), or
2. produce _exactly_ the same answer that AD would produce -- crucially they always return a structural tangent for a non-primitive composite type.

Moreover, assume that users provide structural tangents -- this restriction could be removed by AD authors by shoving a call to `destructure` at the end of their AD if that's something they want to do.

The consequence of the above is that we can safely assume that the input and output of any rule will be either
1. a structural tangent (if the primal is a non-primitive composite type)
2. whatever we have defined its tangent type to be (if it's a primitive type like `Float64` or `Array`).

I'm intentionally not trying to define precisely what a primitive is, but will assume that everyone agrees that `Array`s are an example of a primitive, and that we are happy with representing the tangent of an `Array` by another `Array`.

Assume that structural tangents are a valid representation of a tangent of any non-primitive composite type, convenience for rule-writing aside.

More generally, assume that if an AD successfully runs on a given function under the above assumptions, they give the answer desired (the "correct" answer). Consequently, two core goals of the proposed recipe are to make it
1. easy to never write a rule which prevents an AD from differentiating a programme that they already know how to differentiate,
2. make it easy to write rules using intuitive representations of tangents.





## The Formalism

First consider a specific case -- we'll optimise the implementation later, and provide more general examples in `examples.jl`.

Consider a function `f(x::AbstractArray) -> y::AbstractArray`. Lets assume that there's just one method, so we can be sure that a generic fallback will be hit, regardless the concrete type of the argument.

The intuition behind the recipe is to find a function which is equivalent to `f`, whose rules we know how to write safely. If we can find such a function, AD-ing it will clearly give the correct answer (the same as running AD on `f` itself) -- the following lays out an approach to doing this.

The recipe is:
1. Map `x` to an `Array`, `x_dense`, using `getindex`. Call this operation `destructure` (it's essentially `collect`).
2. Apply `f` to `x_dense` to obtain `y_dense`.
3. Map `y_dense` onto `y`. Call this operation `(::Restructure)`.

I'm going to define equivalence of output structurally -- two `AbstractArray`s are equal if
1. they are primitives, and whatever notion of equality we have for them holds, or
2. they are composite types, and each of their fields are equal under this definition.

The reason for this notion of equality is that AD (as proposed above) treats concrete subtypes of AbstractArray no differently from any other composite type.

A very literal implementation of this for a function like `*` is something like the following:
```julia
function rrule(config::RuleConfig, ::typeof(*), A::AbstractMatrix, B::AbstractMatrix)

    # Produce dense versions of A and B, and the pullbacks of this operation.
    A_dense, destructure_A_pb = rrule(destructure, A)
    B_dense, destructure_B_pb = rrule(destructure, B)

    # Compute dense primal.
    C_dense = A_dense * B_dense

    # Compute structured primal without densifying to ensure that we get structured `C` back
    # if that's what the primal would do.
    C = A * B

    # Construct pullback of Restructure. We generally need to extract some information from
    # C in order to find the structured version.
    _, restructure_C_pb = rrule_via_ad(config, Restructure(C), C_dense)

    function my_mul_generic_pullback(C̄)

        # Recover natural cotangent.
        _, C̄_nat = restructure_C_pb(C̄)

        # Compute pullback using natural cotangent of C.
        Ā_nat = C̄_nat * B_dense'
        B̄_nat = A_dense' * C̄_nat

        # Transform natural cotangents w.r.t. A and B into structural (if non-primitive).
        _, Ā = destructure_A_pb(Ā_nat)
        _, B̄ = destructure_B_pb(B̄_nat)
        return NoTangent(), Ā, B̄
    end

    # The output that we want is `C`, not `C_dense`, so return `C`.
    return C, my_mul_generic_pullback
end
```
I've just written out by hand the rrule for differentiating through the equivalent function.
We'll optimise this implementation shortly to avoid e.g. having to densify primals, and computing the same function twice.
`my_mul` in `examples.jl` verifies the correctness of the above implementation.


`destructure` is quite straightforward to define -- essentially equivalent to `collect`. I'm confident that this is always going to be simple to define, because `collect` is always easy to define.

`Restructure(C)(C_dense)` is a bit trickier. It's the function which takes an `Array` `C_dense` and transforms it into `C`. This feels like a slightly odd thing to do, since we already have `C`, but it's necessary to already know what `C` is in order to construct this function in general -- for example, over-parametrised matrices require this (see the `ScaledMatrix` example in the tests / examples). I'm _reasonably_ confident that this is always going to be possible to define, but I might have missed something.

The PR shows how to implement steps 1 and 3 for `Array`s, `Diagonal`s, `Symmetric`s, and a custom `AbstractArray` `ScaledMatrix`.





## Acceptable (Co)Tangents for `Array`

Any `AbstractArray` is an acceptable (co)tangent for an `Array` (provided it's the right size, and its elements are appropriate (co)tangents for the elements of the primal `Array`).
I'm going to assume this is true, because I can't see any obvious reason why it wouldn't be. 
If anyone feels otherwise, please say.

For example, this means that a `Diagonal{Float64}` is a valid (co)tangent for an `Array{Float64}`.





## Optimising rrules using Natural Pullbacks

The basic example layed out above was very sub-optimal. Consider the following (equivalent) re-write
```julia
function rrule(config::RuleConfig, ::typeof(*), A::AbstractMatrix, B::AbstractMatrix)

    # Produce pullbacks of destructure.
    destructure_A_pb = pullback_of_destructure(config, A)
    destructure_B_pb = pullback_of_destructure(config, B)

    # Primal computation.
    C = A * B

    # Find pullback of restructure.
    restructure_C_pb = pullback_of_restructure(config, C)

    function my_mul_generic_pullback(C̄)

        # Recover natural cotangent.
        C̄_nat = restructure_C_pb(C̄)

        # Compute pullback using natural cotangent of C.
        Ā_nat = C̄_nat * B'
        B̄_nat = A' * C̄_nat

        # Transform natural cotangents w.r.t. A and B into structural (if non-primitive).
        Ā = destructure_A_pb(Ā_nat)
        B̄ = destructure_B_pb(B̄_nat)
        return NoTangent(), Ā, B̄
    end

    return C, my_mul_generic_pullback
end
```
A few observations:
1. All dense primals are gone. In the pullback, they only appeared in places where they can be safely replaced with the primals themselves because they're doing array-like things. `C_dense` appeared in the construction of `restructure_C_pb`, however, we were using a sub-optimal implementation of that function. Much of the time, `restructure_of_pb` doesn't require `C_dense` in order to know what the pullback would look like and, if it does, it can be obtained from `C`.
2. All direct calls to `rrule_via_ad` have been replaced with calls to functions which are defined to returns the things we actually need (the pullbacks). These typically have efficient (and easy to write) implementations.
3. `C̄_nat` could be any old `AbstractArray`. For example, `pullback_of_restructure` for a `Diagonal` returns a `Diagonal`. This is good -- it means we might occassionally get faster computations in the pullback. 

Roughly speaking, the above implementation has only one additional operation than our existing rrules involving `ProjectTo`, which is a call to `restructure_C_pb`, which handles converting a structural tangent for `C̄` into the corresponding natural. Currently we require users to do this by hand, and no clear guidance is provided regarding the correct way to handle this conversion, in contrast to the clarity provided here. In this sense, all that the above is doing is providing a well-defined mechanism by which users can obtain natural cotangents from structural cotangents, so it should ease the burden on rule-implementers.

Almost all of the boilerplate in the above example can be removed by utilising the `wrap_natural_pullback` utility function defined in the PR, as in the example at the top of this note.





## Gotchas

There does seem to be something that goes wrong when primals access non-public fields of types (array authors are obviously allowed to do this), but the generic rrules assume that only the AbstractArray API is used.
I don't think this differs from what we're doing at the minute, so probably we're suffering from this already and just haven't hit it yet.
See the third example in `examples.jl`, involving a `Symmetric`.

This is a particularly interesting case because `parent` is exported from `Base`, effectively making the field of a `Symmetric` part of its public API.

I'm not really sure how to think about this but, as I say, I suspect we're already suffering from it, so I'm not going to worry about it for now.





## Summary

The above lays out a mechanism for writing generic rrules for AbstractArrays, out of which drops what I believe to be a good candidate for a precise definition of the natural (co)tangent of any particular AbstractArray.

There are a lot more examples in `examples.jl` that I would encourage people to work through. Moreover, the `Symmetric` results are a little odd, but I think make sense.
Additionally, implementations of `destructure` and `Restructure` can be moved to the tests because they're really just used to verify the correctness of manually implementations of their pullbacks.

I've presented this work in the context of `AbstractArray`s, but the general scheme could probably be extended to other types by finding other canonical types (like `Array`) on which people's intuition about what ought to happen holds.

The implementation are also limited to arrays of real numbers to avoid the need to recursively apply `destructure` / `restructure`. This restriction could be dropped in practice, and recursive definitions applied.

I'm sure there's stuff above which is unclear -- please let me know if so. There's more to say about a lot of this stuff, but I'll stop here in the interest of keeping this concise.

Please now go and look at `examples.jl`.
