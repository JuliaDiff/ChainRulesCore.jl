"""
    ignore_gradients(f::Function)

Tells the AD system to ignore the gradients of the wrapped closure. The primal computation
(forward pass) is executed normally.

```julia
ignore_gradients() do
    value = rand()
    push!(collection, value)
end
```

Using this incorrectly could lead to incorrect gradients. E.g.
```julia
function wrong_grads(x)
    y = ones(3)
    ignore_gradients() do
        push!(y, x)
    end
    return sum(y)
end
```
"""
ignore_gradients(f::Function) = f()

"""
    ignore_gradients(x)

Tells the AD system to ignore the gradients of the argument. Can be used to avoid
unnecessary computation of gradients.

```julia
ignore_gradient(x) * w
```
"""
ignore_gradients(x) = x

@non_differentiable ignore_gradients(f)

"""
    @ignore_gradients (...)

Tell the AD system to ignore the expression. Equivalent to `ignore_gradients() do (...) end`.
"""
macro ignore_gradients(ex)
    return :(ChainRulesCore.ignore_gradients() do
        $(esc(ex))
    end)
end
