# On writing good `rrule` / `frule` methods

## Use `Zero()` or `One()` as return value

The `Zero()` and `One()` differential objects exist as an alternative to directly returning
`0` or `zeros(n)`, and `1` or `I`.
They allow more optimal computation when chaining pullbacks/pushforwards, to avoid work.
They should be used where possible.

## Use `Thunk`s appropriately

If work is only required for one of the returned differentials, then it should be wrapped in a `@thunk` (potentially using a `begin`-`end` block).

If there are multiple return values, their computation should almost always be wrapped in a `@thunk`.

Do _not_ wrap _variables_ in a `@thunk`; wrap the _computations_ that fill those variables in `@thunk`:

```julia
# good:
∂A = @thunk(foo(x))
return ∂A

# bad:
∂A = foo(x)
return @thunk(∂A)
```
In the bad example `foo(x)` gets computed eagerly, and all that the thunk is doing is wrapping the already calculated result in a function that returns it.

Do not use `@thunk` if this would be equal or more work than actually evaluating the expression itself.
Examples being:
- The expression being a constant
- The expression is merely wrapping something in a `struct`, such as `Adjoint(x)` or `Diagonal(x)`
- The expression being itself a `thunk`
- The expression being from another `rrule` or `frule`;
  it would be `@thunk`ed if required by the defining rule already.
- There is only one derivative being returned, so from the fact that the user called
  `frule`/`rrule` they clearly will want to use that one.

## Code Style

Use named local functions for the `pullback` in an `rrule`.

```julia
# good:
@rrule function rrule(::typeof(foo), x)
    Y = foo(x)
    function foo_pullback(x̄)
        return NO_FIELDS, bar(x̄)
    end
    return Y, foo_pullback
end
#== output
julia> rrule(foo, 2)
(4, var"#foo_pullback#11"())
==#

# bad:
@rrule function rrule(::typeof(foo), x)
    return foo(x), x̄ -> (NO_FIELDS, bar(x̄))
end
#== output:
julia> rrule(foo, 2)
(4, var"##9#10"())
==#
```

While this is more verbose, it ensures that if an error is thrown during the `pullback` the [`gensym`](https://docs.julialang.org/en/v1/base/base/#Base.gensym) name of the local function will include the name you gave it.
This makes it a lot simpler to debug from the stacktrace.

## Write tests

In [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl)
there are fairly decent tools for writing tests based on [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).
Take a look at existing [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) tests and you should see how to do stuff.

!!! warning
    Use finite differencing to test derivatives.
    Don't use analytical derivations for derivatives in the tests.
    Those are what you use to define the rules, and so can not be confidently used in the test.
    If you misread/misunderstood them, then your tests/implementation will have the same mistake.

## CAS systems are your friends.

It is very easy to check gradients or derivatives with a computer algebra system (CAS) like [WolframAlpha](https://www.wolframalpha.com/input/?i=gradient+atan2%28x%2Cy%29).
