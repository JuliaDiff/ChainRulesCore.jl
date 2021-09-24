"""
    ignore_gradients() do
        ...
    end

A convenience function that tells the AD system to ignore the gradients of the wrapped
closure. The forward pass is executed normally.

Using this incorrectly, for example by including calculations in the closure, could lead
to incorrect gradients. A possible use case is logging of quantities in the forward pass.
"""
ignore_gradients(f) = f()

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
