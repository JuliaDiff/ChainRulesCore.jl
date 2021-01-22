# Design Notes: Why can you change the primal?
#TODO generalize intro/title to ask why `rrule` is how it is, and in particular about changing primal

These design notes are to help you understand why ChainRules allows the primal computation, to be changed.
We will focus this discussion on reverse mode and `rrule`, though the same also applies to forwards mode and `frule`.
In fact, it has a particular use in forward mode for efficiently calculating the pushforward of a differential equation solve via expanding the system of equations to also include the derivatives, and solving all at once.
In forward mode it is related to the fusing of `frule` and `pushforward`.
In reverse mode we can focus on the the distinct primal and gradient passes.


## The Journey to `rrule`

Let's imagine a different system for rules, one that doesn't let you do this.
This system is what a lot of AD systems have --- it is what [Nabla.jl](https://github.com/invenia/Nabla.jl/)[^1] had originally.
We will have a primal (i.e. forward) pass that directly executes the primal function and just records its _inputs_ and its _output_ (as well as the _primal function_ itself) onto the tape.[^2].
Then during the gradient (i.e. reverse) pass it has a function which receives those records from the tape along with the sensitivity of the output, and gives back the sensitivity of the input.
We will call this function `pullback_at`, as it pulls back the sensitivity at a given primal point.
To make this concrete:
```julia
y = f(x)  # primal program
x̄ = pullback_at(f, x, y, ȳ)
```

Let's write one:
```julia
y = sin(x)
pullback_at(::typeof(sin), x, y, ȳ) = ȳ * cos(x)
```

Great. So far so good.
As a short exercise the reader might like to implement the one for the [logistic sigmoid](https://en.wikipedia.org/wiki/Logistic_function#Derivative).
It also works without issue.
#TODO should I actually just include logistic sigmoid a second example though out, and write its code next the the code for `sin`?


Now let's consider why we implement `rrules` like this in the first place.
One key reason, [^3] is to allow us to insert our domain knowledge to do better than the AD would do just by breaking everything down into `+` and `*` etc.
What insights do we have about `sin` and `cos`?
Here is one:
```julia
julia> @btime sin(x) setup=(x=rand());
  3.838 ns (0 allocations: 0 bytes)

julia> @btime cos(x) setup=(x=rand());
  4.795 ns (0 allocations: 0 bytes)

julia> 3.838 + 4.795
8.633

julia> @btime sincos(x) setup=(x=rand());
  6.028 ns (0 allocations: 0 bytes)
```
It is \~30%[^4] faster to compute `sin` and `cos` at the same time via `sincos` than it is to compute them one after the other.
How can we incorporate this insight into our system?
We know that we can compute the `cos(x)` at the same time as the `sin(x)` is computed in primal, because it only depends on `x` --- we don't need to know `ȳ`.
but there is no-where to put it that is accessible both to the primal pass, and the gradient pass code.

What if we introduced some variable called `intermediates` that is also record onto the tape during the primal pass?
We would need to be able to modify the primal pass to do this, so we can actually put the data into the `intermediates`.
So we will introduce a function: `augmented_primal`, that will return the primal output, plus the `intermediates` that we want to reuse in the gradient pass.
Then we will make our AD system replace calls to the primal with calls to the `augmented_primal` of the primal function; and take care of all the bookkeeping.
So that would look like:
```julia
y = f(x)  # primal program
y, intermediates = augmented_primal(f, x)
x̄ = pullback_at(f, x, y, ȳ, intermediates)
```

Let's try writing this now:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, (; cx=cx)  # use a NamedTuple for the intermediates
end

pullback_at(::typeof(sin), x, y, ȳ, intermediates) = ȳ * intermediates.cx
```

Cool!
That lets us do what we wanted.
We net deceased the time it takes to run the primal and gradient passes.
We have now demonstrated the title question of why we needed to be able to modify the primal pass.
We will go into that more later, and have some more examples of use.
But first lets continue to see how we go from that `augmented_primal` _ `pullback_at` to [`rrule`](@ref).


One thing we notice when looking at `pullback_at` is it really is starting to have a lot of arguments.
It had a fair few already, and now we are adding `intermediates` as well.
Not to mention this is a fairly simple function, only 1 input, no keyword arguments.
Furthermore, we don't even use all of them all the time.
The original new code for pulling back `sin` no longer needs the `x`, and it never needed `y` (though `sigmoid` does).
Having all this extra stuff means that the `pullback_at` function signature is long, and we are putting a bucnh of extra stuff on the tape, using more memory.
What if we generalized the idea of the `intermediate` named tuple, and had a struct that just held anything we might want put on the tape.
```julia
struct PullbackMemory{P, S}
  primal_function::P
  state::S
end
# convenience constructor:
Memory(primal_function; state...) = PullbackMemory(primal_function, state)
# convenience accessor so that `m.x` is same as `m.state.x`
Base.getproperty(m::PullbackMemory, propname) = getproperty(getfield(m, :state), propname)
```

So changing our API we have:
```julia
y = f(x)  # primal program
y, pb = augmented_primal(f, x)
x̄ = pullback_at(pb, ȳ)
```
which is much cleaner.

Let's try writing with this now:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end

pullback_at(pb::PullbackMemory{typeof(sin)}, ȳ) = ȳ * pb.cx
```

I think that looks pretty nice.

One way we could make it look a bit nicer for using, is if the `PullbackMemory` was actually a callable object.
since `pullback_at` only has the 2 arguments,
and conceptually the `PullbackMemory` is a more fixed thing -- it is fully determined by the end of the primal pass,
then during the reverse pass the argument `ȳ` is successively computed.
We can do that via call overloading:
```julia
y = f(x)  # primal program
y, pb = augmented_primal(f, x)
x̄ = pb(ȳ)
```
and for `sin`:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end
(pb::PullbackMemory)(ȳ) = ȳ * pb.cx
```

Those looking closely will spot what we have done here.
We now have a object (`pb`) that acts on the cotangent of the output of the primal (`ȳ`) to give us the cotangent of the input of the primal function (`x̄`).
_`pb` is not just the **memory** of state required for the `pullback`, it **is** the pullback._

We have one final thing to do.
Lets think about make the code easy to modify.
Lets go back and think about the changes we would have to make to swap to using `sincos` from our original `sin` then `cos` way of writing.
To rewrite that original formulation in the new pullback form we have:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y = sin(x)
  return y, PullbackMemory(sin; x=x)
end
(pb::PullbackMemory)(ȳ) = ȳ * cos(pb.x)
```
To go from that to:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  return y, PullbackMemory(sin; cx=cx)
end
(pb::PullbackMemory)(ȳ) = ȳ * pb.cx
```
we need to make a series of changes.
We need to update what work is done in the primal to compute `cx`.
We need to update what was stored in the `PullbackMemory`.
And we need to update the the function that applies the pullback so it uses the new thing that was stored.
It's important these thing all stay in sync.
It's not to bad for this simple example with just 1 thing to remember.
For more complicated multi-argument functions, like will be talked about below, you often end up needing to remember half a dozen things, like sizes and indices relating to each input/output.
So it gets a little more fiddly to make sure you remember all the things you need to a and give them same name in both places.
_Is there a way we can automatically just have all the things we use remembered for us?_

Surprisingly for such a specific request, there actually is.
This is a closure.
A closure in julia is a callable structure, that automatically contains a field for every object from its parent scope that is used in its body.
There are [incredible ways to abuse this](https://invenia.github.io/blog/2019/10/30/julialang-features-part-1#closures-give-us-classic-object-oriented-programming); but here we can in-fact use closures exactly as they are intended.
Replacing `PullbackMemory` with a closure that works the same lets us avoid manually controlling what is remembers, _and_ lets us avoid writing separately the call overload.
So we have for `sin`:
```julia
y = sin(x)
function augmented_primal(::typeof(sin), x)
  y, cx = sincos(x)
  pb = ȳ -> cx * ȳ  # pullback closure. closes over `cs`
  return y, pb
end
```

Our `augmented_primal` is now with-in spitting distance of `rrule`.
All that is left is a rename, and some extra conventions around multiple outputs and gradients with respect to callable objects.

This has been a journey into how we get to [`rrule`](@ref) as it is defined in `ChainRulesCore`.
We started with anu unaugmented primal function and a `pullback_at` function that only saw the inputs and outputs of the primal.
We realized a key limitation of this was that we couldn't share computational work between the primal and and gradient passes.
To solve this we introduced the notation of a some shared `intermediate` that is shared from the primal to the pullback.
We successively improved that idea, first by making it a type that held everything that is needed for the pullback: the `PullbackMemory`.
Which we then made callabled --- so it  was itself the pullback._
Finally, we replaced that seperate callable structure with a closure, which kept everything in one place and made it more convenient.

## More Shared State
TODO



[^1]:
    I am not just picking on Nabla randomly.
    Many of the core developers of ChainRules worked on Nabla prior.
    It's a good AD, but ChainRules incorporates lessons learned from working on Nabla.

[^2]: which may be an explicit tape, or an implicit tape that is actually incorporated into generated code (ala Zygote)

[^3]:
    Another key reason is if the operations is a primitive that is not defined in terms of more basic operations.
    In many languages this is the case for `sin`; where the actual implementation is in some separate `libm.so`.
    But actually [`sin` in Julia is defined in terms of a polynomial](https://github.com/JuliaLang/julia/blob/caeaceff8af97565334f35309d812566183ec687/base/special/trig.jl).
    It's fairly vanilla julia code.
    It shouldn't be too hard for an AD that only knows about basic operations like `+` and `*` to AD through it.
    Though that will incur something that looks a lot like truncation error (in apparent violation of Griewank and Walther's 0th Rule of AD).
    In anycase, that is another discussion, for another day.

[^4]:
Sure, this is small-fries and depending on julia version might just get solved by the optimizer[^5], but go with it for the sake of example.

[^5]:
  To be precise this is very likely to be solved by the optimizer inlining both and then performing common subexpression elimination, with the result that it generates the code for `sincos` just form having `sin` and `cos` inside the same function.
  However, this actually doesn't apply in the case of AD as it is not possible to inline code called in the gradient pass into the primal pass -- those are seperate functions called at very different times.
  This is something [opaque closures](https://github.com/JuliaLang/julia/pull/37849) should help solve.
