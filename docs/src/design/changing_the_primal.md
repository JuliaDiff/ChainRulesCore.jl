# Design Notes: Why can you change the primal?
These design notes are to help you understand why ChainRules allows the primal computation, to be changed.
We will focus this discussion on reverse mode and `rrule`, though the same also applies to forwards mode and `frule`.
In fact, it has a particular use in forward mode for efficiently calculating the pushforward of a differential equation solve via expanding the system of equations to also include the derivatives, and solving all at once.
In forward mode it is related to the fusing of `frule` and `pushforward`.
In reverse mode we can focus on the the distinct primal and gradient passes.


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
We have now demostrated the title question of why we needed to be able to modify the primal pass.
We will go into that more later, and have some more examples of use.
But first lets continue to see how we go from that `augmented_primal` _ `pullback_at` to [`rrule`](@ref).


One thing we notice when looking at

# TODO:
 - Highlight that there is a bunch of arguments to `pullback_at`.
    - What if we wrapped them all up in a structure, like we did for `intermidates`?
 - since `pullback_at` takes only 1 argument, other than `ȳ` so it would elegant if we make that struct callable, and had it do `pullback_at`
 - It seems not geat that we are  including a bunch of things like the `output` and the `input` as well as `intermidiates`, why not treat them just like `intermidates` and only include the ones we are going to use?
 - Specifying them is kind of annoying, as is keeping them in-sync between `augmented_primal` and `pullback_at`.
    - solve by making it a closure




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
