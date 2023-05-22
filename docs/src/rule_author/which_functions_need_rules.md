# Which functions need rules?

In principle, a perfect AD system only needs rules for basic operations and can infer the rules for more complicated functions automatically.
In practice, performance needs to be considered as well.

Some functions use `ccall` internally, for example [`^`](https://github.com/JuliaLang/julia/blob/v1.5.3/base/math.jl#L886).
These functions cannot be differentiated through by AD systems, and need custom rules.

Other functions can in principle be differentiated through by an AD system, but there exists a mathematical insight that can dramatically improve the computation of the derivative.
An example is numerical integration, where writing a rule implementing the [fundamental theorem of calculus](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus) removes the need to perform AD through numerical integration.

Furthermore, AD systems make different trade-offs in performance due to their design.
This means that a certain rule will help one AD system, but not improve (and also not harm) another.
Below, we list some patterns relevant for the [Zygote.jl](https://github.com/FluxML/Zygote.jl) AD system.

Rules for functions which mutate its arguments, e.g. `sort!`, should not be written at the moment.
While technically they are supported, they would break [Zygote.jl](https://github.com/FluxML/Zygote.jl) such that [it would sometimes quietly return the wrong answer](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/242).
This may be resolved in the future by [allowing AD systems to opt-in or opt-out of certain types of rules](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/270).

### Patterns that need rules in [Zygote.jl](https://github.com/FluxML/Zygote.jl)

There are a few classes of functions that Zygote cannot differentiate through.
Custom rules will need to be written for these to make AD work.

Other patterns can be AD'ed through, but the backward pass performance can be greatly improved by writing a rule.

#### Functions which mutate arrays
For example,
```julia
function addone(a::AbstractArray)
    b = similar(a)
    b .= a .+ 1
    return sum(b)
end
```
complains that
```julia
julia> using Zygote
julia> gradient(addone, a)
ERROR: Mutating arrays is not supported
```
However, upon adding the `rrule` (restart the REPL after calling `gradient`)
```julia
function ChainRules.rrule(::typeof(addone), a)
    y = addone(a)
    function addone_pullback(ȳ)
        return NoTangent(), ones(length(a))
    end
    return y, addone_pullback
end
```
the gradient can be evaluated:
```julia
julia> gradient(addone, a)
([1.0, 1.0, 1.0],)
```
Notice that `addone(a)` mutates another array `b` internally, but **not** its input.
This is commonly done in less trivial functions, and is often what Zygote's `Mutating arrays is not supported` error is telling you,
even though you did not intend to mutate anything.
Functions which mutate their own input are much more problematic.
These are the ones named (by convention) with an exclamation mark, such as `fill!(a, x)` or `push!(a, x)`.
It is not possible to write rules which handle all uses of such a function correctly, on current Zygote.


!!! note "Why restarting REPL after calling `gradient`?"
    When `gradient` is called in `Zygote` for a function with no `rrule` defined, a backward pass for the function call is generated and cached.
    When `gradient` is called for the second time on the same function signature, the backward pass is reused without checking whether an an `rrule` has been defined between the two calls to `gradient`.
    
    If an `rrule` is defined before the first call to `gradient` it should register the rule and use it, but that prevents comparing what happens before and after the `rrule` is defined.
    To compare both versions with and without an `rrule` in the REPL simultaneously, define a function `f(x) = <body>` (no `rrule`), another function `f_cr(x) = f(x)`, and an `rrule` for `f_cr`.

    Calling `Zygote.refresh()` will often have the same effect as restarting the REPL.

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
    function exception_pullback(ȳ)
        return NoTangent(), 2*x
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
function mse(y, ŷ)
    N = length(y)
    s = 0.0
    for i in 1:N
        s +=  (y[i] - ŷ[i])^2.0
    end
    return s/N
end
```
takes a lot longer to AD through
```julia
julia> y = rand(30)
julia> ŷ = rand(30)
julia> @btime gradient(mse, $y, $ŷ)
  38.180 μs (993 allocations: 65.00 KiB)
```
than if we supply an `rrule`, (restart the REPL after calling `gradient`)
```julia
function ChainRules.rrule(::typeof(mse), x, x̂)
    output = mse(x, x̂)
    function mse_pullback(ȳ)
        N = length(x)
        g = (2 ./ N) .* (x .- x̂) .* ȳ
        return NoTangent(), g, -g
    end
    return output, mse_pullback
end
```
which is much faster
```julia
julia> @btime gradient(mse, $y, $ŷ)
  143.697 ns (2 allocations: 672 bytes)
```

#### In-place accumulation

In-place accumulation of gradients is slow in `Zygote`.
The issue, demonstrated in the following example, is that the gradient of `getindex` allocates an array of zeros with a single non-zero element. 
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
    function sum3_pullback(ȳ)
        grad = zeros(length(a))
        grad[1:3] .+= ȳ
        return NoTangent(), grad
    end
    return y, sum3_pullback
end
```
turns out to be significantly faster 
```julia
julia> @btime gradient(sum3, rand(30))
  192.818 ns (3 allocations: 784 bytes)
```
