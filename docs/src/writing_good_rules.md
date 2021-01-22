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
function rrule(::typeof(foo), x)
    Y = foo(x)
    function foo_pullback(Ȳ)
        return NO_FIELDS, bar(Ȳ)
    end
    return Y, foo_pullback
end
#== output
julia> rrule(foo, 2)
(4, var"#foo_pullback#11"())
==#

# bad:
function rrule(::typeof(foo), x)
    return foo(x), x̄ -> (NO_FIELDS, bar(x̄))
end
#== output:
julia> rrule(foo, 2)
(4, var"##9#10"())
==#
```

While this is more verbose, it ensures that if an error is thrown during the `pullback` the [`gensym`](https://docs.julialang.org/en/v1/base/base/#Base.gensym) name of the local function will include the name you gave it.
This makes it a lot simpler to debug from the stacktrace.

## Use rule definition tools

Rule definition tools can help you write more `frule`s and the `rrule`s with less lines of code.

### [`@non_differentiable`](@ref)

For non-differentiable functions the [`@non_differentiable`](@ref) macro can be used.
For example, instead of manually defining the `frule` and the `rrule` for string concatenation `*(String..)`, the macro call
```julia
@non_differentiable *(String...)
```
defines the following `frule` and `rrule` automatically
```julia
function ChainRulesCore.frule(var"##_#1600", ::Core.Typeof(*), String::Any...; kwargs...)
    return (*(String...; kwargs...), DoesNotExist())
end
function ChainRulesCore.rrule(::Core.Typeof(*), String::Any...; kwargs...)
    return (*(String...; kwargs...), function var"*_pullback"(_)
        (Zero(), ntuple((_->DoesNotExist()), 0 + length(String))...)
    end)
end
```
Note that the types of arguments are propagated to the `frule` and `rrule` definitions.
This is needed in case the function differentiable for some but not for other types of arguments.
For example `*(1, 2, 3)` is differentiable, and is not defined with the macro call above.

### [`@scalar_rule`](@ref)

For functions involving only scalars, i.e. subtypes of `Number` (no `struct`s, `String`s...), both the `frule` and the `rrule` can be defined using a single [`@scalar_rule`](@ref) macro call. 

Note that the function does not have to be $\mathbb{R} \rightarrow \mathbb{R}$.
In fact, any number of scalar arguments is supported, as is returning a tuple of scalars.

See docstrings for the comprehensive usage instructions.
## Write tests

[ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl)
provides tools for writing tests based on [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).
Take a look at the documentation or the existing [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) tests to see how to write the tests.

!!! warning
    Don't use analytical derivations for derivatives in the tests.
    Those are what you use to define the rules, and so can not be confidently used in the test.
    If you misread/misunderstood them, then your tests/implementation will have the same mistake.
    Use finite differencing methods instead, as they are based on the primal computation.

## CAS systems are your friends.

It is very easy to check gradients or derivatives with a computer algebra system (CAS) like [WolframAlpha](https://www.wolframalpha.com/input/?i=gradient+atan2%28x%2Cy%29).

## Which functions need rules?

In principle, a perfect AD system only needs rules for basic operations and can infer the rules for more complicated functions automatically.
In practice, performance needs to be considered as well.

Some functions use `ccall` internally, for example [`^`](https://github.com/JuliaLang/julia/blob/v1.5.3/base/math.jl#L886).
These functions can not be differentiated through by AD systems, and need custom rules.

Other functions can in principle be differentiated through by an AD system, but there exists a mathematical insight that can dramatically improve the computation of the derivative.
An example is numerical integration, where writing a rule removes the need to perform AD through numerical integration.

Furthermore, AD systems make different trade-offs in performance due to their design.
This means that a certain rule will help one AD system, but not improve (and also not harm) another.
Below, we list some patterns relevant for the [Zygote.jl](https://github.com/FluxML/Zygote.jl) AD system.

### Patterns that need rules in [Zygote.jl](https://github.com/FluxML/Zygote.jl)

There are a few classes of functions that Zygote can not differentiate through.
Custom rules will need to be written for these to make AD work.

Other patterns can be AD'ed through, but the backward pass performance can be greatly improved by writing a rule.

#### Functions which mutate arrays
For example,
```julia
function addone!(array)
    array .+= 1
    return sum(array)
end
```
complains that
```julia
julia> using Zygote
julia> gradient(addone!, a)
ERROR: Mutating arrays is not supported
```
However, upon adding the `rrule` (restart the REPL after calling `gradient`)
```julia
function ChainRules.rrule(::typeof(addone!), a)
    y = addone!(a)
    function addone!_pullback(ȳ)
        return NO_FIELDS, ones(length(a))
    end
    return y, addone!_pullback
end
```
the gradient can be evaluated:
```julia
julia> gradient(addone!, a)
([1.0, 1.0, 1.0],)
```

!!! note "Why restarting REPL after calling `gradient`?"
    When `gradient` is called in `Zygote` for a function with no `rrule` defined, a backward pass for the function call is generated and cached.
    When `gradient` is called for the second time on the same function signature, the backward pass is reused without checking whether an an `rrule` has been defined between the two calls to `gradient`.
    
    If an `rrule` is defined before the first call to `gradient` it should register the rule and use it, but that prevents comparing what happens before and after the `rrule` is defined.
    To compare both versions with and without an `rrule` in the REPL simultaneously, define a function `f(x) = <body>` (no `rrule`), another function `f_cr(x) = f(x)`, and an `rrule` for `f_cr`.

#### Exception handling

Zygote does not support differentiating through `try`/`catch` statements.
For example, differentiating through
```julia
function exception(x)
    try
        return x^2
    catch e
        println("could not square input")
        throw(e)
    end
end
```
does not work
```julia
julia> gradient(exception, 3.0)
ERROR: Compiling Tuple{typeof(exception),Int64}: try/catch is not supported.
```
without an `rrule` defined (restart the REPL after calling `gradient`)
```julia
function ChainRulesCore.rrule(::typeof(exception), x)
    y = exception(x)
    function exception_pullback(ȳ)
        return NO_FIELDS, 2*x
    end
    return y, exception_pullback
end
```

```julia
julia> gradient(exception, 3.0)
(6.0,)
```


#### Loops

Julia runs loops fast.
Unfortunately Zygote differentiates through loops slowly.
So, for example, computing the mean squared error by using a loop
```julia
function mse(y, ŷ)
    N = length(y)
    s = 0.0
    for i in 1:N
        s +=  (y[i] - ŷ[i])^2.0
    end
    return s/N
end
```
takes a lot longer to AD through
```julia
julia> y = rand(30)
julia> ŷ = rand(30)
julia> @btime gradient(mse, $y, $ŷ)
  38.180 μs (993 allocations: 65.00 KiB)
```
than if we supply an `rrule`, (restart the REPL after calling `gradient`)
```julia
function ChainRules.rrule(::typeof(mse), x, x̂)
    output = mse(x, x̂)
    function mse_pullback(ȳ)
        N = length(x)
        g = (2 ./ N) .* (x .- x̂) .* ȳ
        return NO_FIELDS, g, -g
    end
    return output, mse_pullback
end
```
which is much faster
```julia
julia> @btime gradient(mse, $y, $ŷ)
  143.697 ns (2 allocations: 672 bytes)
```

#### Inplace accumulation

Inplace accumulation of gradients is slow in `Zygote`.
The issue, demonstrated in the folowing example, is that the gradient of `getindex` allocates an array of zeros with a single non-zero element. 
```julia
function sum3(array)
    x = array[1]
    y = array[2]
    z = array[3]
    return x+y+z
end
```
```julia
julia> @btime gradient(sum3, rand(30))
  424.510 ns (9 allocations: 2.06 KiB)
```
Computing the gradient with only a single array allocation using an `rrule` (restart the REPL after calling `gradient`)
```julia
function ChainRulesCore.rrule(::typeof(sum3), a)
    y = sum3(a)
    function sum3_pullback(ȳ)
        grad = zeros(length(a))
        grad[1:3] .+= 1.0
        return NO_FIELDS, grad
    end
    return y, sum3_pullback
end
```
turns out to be significantly faster 
```julia
julia> @btime gradient(sum3, rand(30))
  192.818 ns (3 allocations: 784 bytes)
```

