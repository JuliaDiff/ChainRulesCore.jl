# Tips for making your package work with AD

## Ignoring gradients for certain expressions

There exists code that is not meant to be differentiated through, for example logging.
In some cases, AD systems might work perfectly well with that code, but in others they might not.
A convenience function `ignore_derivatives` is provided to get around this issue.
It captures the functionality of both `Zygote.ignore` and `Zygote.dropgrad`.

For example, Zygote does not support mutation, so it will break if you try to store intermediate values as in the following example:
```julia
results = []
things = []

function loss(x, y)
    result = f(x, y)
    thing = g(x)
    
    # log
    push!(results, result)
    push!(things, thing)

    return result + thing
end
```

It is possible to get around this by using the `ignore_derivatives` function
```julia
ignore_derivatives() do
    push!(results, result)
    push!(things, thing)
end
```
or using a macro for one-liners
```julia
@ignore_derivatives push!(things, thing)
```

It is also possible to use this on individual objects, e.g.
```julia
ignore_derivatives(a) + b
```
will ignore the gradients for `a` only.

Passing in instances of functors (callable structs), `ignore_derivatives(functor)`, will make them behave like normal structs, i.e. propagate without being called and dropping their gradients.
If you want to call a functor in the primal computation, wrap it in a closure: `ignore_derivatives(() -> functor())`
