"""
    ignore() do
        ...
    end

Tell ChainRulesCore to ignore a block of code. Everything inside the `do`
block will run on the forward pass as normal, but ChainRulesCore won't try to
differentiate it at all. This can be useful for e.g. code that does logging of
the forward pass.

Obviously, you run the risk of incorrect gradients if you use this incorrectly
"""
ignore(f) = f()

"""
    @ignore (...)

Tell ChainRulesCore to ignore an expression. Equivalent to
`ignore() do (...) end`.
Example:

```julia-repl
julia> f(x) = x
julia> _, v_pullback = ChainRulesCore.rrule(ChainRulesCore.ignore, f)
```
"""

function frule((_, ḟ), ::typeof(ignore), f)
    return f(), Zero()
end

function rrule(::typeof(ignore), f)
    function ignore_pullback(ȳ)
        return (NO_FIELDS, Zero())
    end
    return f(), ignore_pullback
end

macro ignore(ex)
    return :(ChainRulesCore.ignore() do 
        $(esc(ex))
    end)
end
