# Converting ZygoteRules.@adjoint to `rrule`s

[ZygoteRules.jl](https://github.com/FluxML/ZygoteRules.jl) is a legacy package similar to ChainRulesCore but supporting [Zygote.jl](https://github.com/FluxML/Zygote.jl) only.

If you have some rules written with ZygoteRules it is a good idea to upgrade them to use ChainRules instead.
Zygote will still be able to use them, but so will other AD systems,
and you will get access to some more advanced features.
Some of these features are currently ignored by Zygote, but could be supported in the future.

## Example
Consider the function
```julia
struct Foo
    a::Float64,
    b::Float64
end

f(x, y::Foo, z) = 2*x + y.a
```

### ZygoteRules
```julia
@adjoint function f(x, y::Foo, z)
    f_pullback(Ω̄) = (2Ω̄, NamedTuple(;a=Ω̄, b=nothing), nothing)
    return f(x, y, z), f_pullback
end
```

### ChainRules
```julia
function rrule(::typeof(f), x, y::Foo, z)
    f_pullback(Ω̄) = (NoTangent(), 2Ω̄, Tangent{Foo}(;a=Ω̄), ZeroTangent())
    return f(x, y, z), f_pullback
end
```

## Write as a `rrule(::typeof(f), ...)`
No magic macro here, `rrule` is the function that it is.
The function it is the rule for is the first argument, or second argument if you need to take a [`RuleConfig`](@ref).

Note that when writing the rule for constructor you will need to use `::Type{Foo}`, not `typeof(Foo)`.
See docs on [Constructors](@ref).

## Include the derivative with respect to the function object itself
The `ZygoteRules.@adjoint` macro automagically[^1] inserts an extra `nothing` in the return for the function it generates to represent the derivative of output with respect to the function object.
ChainRules as a philosophy avoids magic as much as possible, and thus require you to return it explicitly.
If it is a plain function (like `typeof(sin)`), then the tangent will be [`NoTangent`](@ref).


[^1]: unless you write it in functor form (i.e. `@adjoint (f::MyType)(args...)=...`), in that case like for `rrule` you need to include it explicitly.

## Tangent Type changes
ChainRules uses tangent types that must represent vector spaces (i.e. tangent spaces).
They need to have things like `+` defined on them.
ZygoteRules takes a more adhoc approach to this.

### `nothing` becomes an `AbstractZero`
ZygoteRules uses nothing to represent some sense of zero, in a primal type agnostic way.
There are many senses of zero.
ChainRules represents two of them, as subtypes of [`AbstractZero`](@ref).

[`ZeroTangent`](@ref) for the case that there is no relationship between the primal output and the primal input.
[`NoTangent`](@ref) for the case where conceptually the tangent space doesn't exist.
e.g. what is the Tangent to a String or an index: those can't be perturbed.

See [FAQ on the difference between `ZeroTangent` and `NoTangent`](@ref faq_abstract_zero).
At the end of the day it doesn't matter too much if you get them wrong.
`NoTangent` and `ZeroTangent` more or less act identically.

### `Tuple`s and `NamedTuple`s become `Tangent{T}`s
Zygote uses `Tuple`s and `NamedTuple`s to represent the structural tangents for `Tuple`s and `struct`s respectively.
ChainRules core provides a generic [`Tangent{T}`](@ref Tangent) to represent the structural tangent of a primal type `T`.
It takes positional arguments if representing tangent for a `Tuple`.
Or keyword argument to represent the tangent for a `struct` or a `NamedTuple`.
When representing a `struct` you only need to list the nonzero fields -- any not given are implicit considered to be [`ZeroTangent`](@ref).

When we say structural tangent we mean tangent types that are based only on the structure of the primal.
This is in contrast to a natural tangent which captures some knowledge based on what the primal type represents.
(E.g. for arrays a natural tangent is often the same kind of array).
For more details see the the [design docs on the many tangent types](@ref manytypes)


## Calling back into AD (`ZygoteRules.pullback`)
Rules that need to call back into the AD system, e.g, for higher order functions like `map(f, xs)`, need to be changed.
In `ZygoteRules` you can use `ZygoteRules.pullback` or `ZygoteRules._pullback`, which will always result in calling into Zygote.
Since ChainRules is AD agnostic, you can't do that.
Instead you use a [`RuleConfig`](@ref) to specify requirements of an AD system e.g `::RuleConfig{>:HasReverseMode}` work for Zygote,
and then use [`rrule_via_ad`](@ref).

See the [docs on calling back into AD](@ref config) for more details.

## Consider adding some thunks

A feature ChainRulesCore offers that ZygoteRules doesn't is support for thunks.
Thunks delay work until it is needed, and avoid it if it never is.
See docs on [`@thunk`](@ref), [`Thunk`](@ref), [`InplaceableThunk`](@ref).

You don't have to use thunks, though.
It is easy to go overboard with using thunks.

## Testing Changes

One of the advantages of using ChainRules is that you can easily and robustly test your rules with [ChainRulesTestUtils.jl](https://juliadiff.org/ChainRulesTestUtils.jl/stable/).
This uses finite differencing to test the accuracy of derivative, as well as checks the correctness of the API.
It should catch anything you might have gotten wrong referred to in this page.

The test for the above example is `test_rrule(f, 2.5, Foo(9.9, 7.2), 31.0)`.
You can see it looks a lot like an example call to `rrule`, just with the prefix `test_` added to the start.

## `@nograd` becomes `@non_differentiable`
Probably more or less with no changes.
[`@non_differentiable`](@ref) also lets you specify a signature in case you want to restrict non-differentiability to a certain subset of argument types.

## No such thing as `literal_getproperty`
That is just `getproperty`, it takes `Symbol`.
It should constant-fold.
It likely doesn't though as Zygote doesn't play nice with the optimizer.

## Take embedded spaces and types seriously
Traditionally Zygote has taken a very laissez-faire attitude towards types and mathematical spaces.
Sometimes treating `Real`s as embedded in the `Complex` plane; sometimes not.
Sometimes treating sparse and structuredly-sparse matrix as embedded in the space of dense matrices.
Writing rules that apply to any `Array{T}` which perhaps are only applicable for `Array{<:Real}` and not so much for `Array{Quaternion}`.
Traditionally ChainRules takes a much more considered approach.

See for example our [docs on how to handle complex numbers](@ref complexfunctions) correctly.
(The outcome of several long long long discussions with a number of experts in our community)

Now, I am not here to tell you what to do in your package, but this is a good time to reconsider how seriously you take these things in the rules you are converting.

## What if I miss something

It is not great, but it probably OK.
Zygote's ChainRules interface is fairly forgiving.
Other AD systems may not be.
If you test with [ChainRulesTestUtils.jl](https://juliadiff.org/ChainRulesTestUtils.jl/stable/) then you can be confident that you didn't miss anything.
