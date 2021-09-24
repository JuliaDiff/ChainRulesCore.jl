# Tips for making your package work with AD

## Ignoring gradients for certain expressions

There exists code that is not meant to be differentiated through, for example logging.
In some cases, AD systems might work perfectly well with that code, but in others they might not.
A convenience function `ignore_gradients` is provided to get around this issue.
It captures the functionality of both `Zygote.ignore` and `Zygote.dropgrad`

For example, Zygote will break if you try to store intermediate values like so because it does not support mutation.
```julia
somes = []
things = []

function loss(x, y)
    some = f(x, y)
    thing = g(x)
    
    # log
    push!(somes, some)
    push!(things, thing)

    return some + thing
end
```

It is possible to get around this by using the `ignore_gradients` function
```julia
ignore_gradients() do
    push!(somes, some)
    push!(things, thing)
end
```
or using a macro for one-liners
```julia
@ignore_gradients push!(things, thing)
```

It is also possible to use this on individual objects, e.g.
```julia
ignore_gradients(a) + b
```
will ignore the gradients for `a` only.
