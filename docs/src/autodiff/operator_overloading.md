# Operator Overloading

## Examples

### ForwardDiffZero

````@eval
using Markdown
code = read(joinpath(@__DIR__,"../../../test/demos/forwarddiffzero.jl"), String)
code = replace(code, raw"$" => raw"\$")
Markdown.parse("""
```julia
$(code)
```
""")
````

### ReverseDiffZero

````@eval
using Markdown
code = read(joinpath(@__DIR__,"../../../test/demos/reversediffzero.jl"), String)
code = replace(code, raw"$" => raw"\$")
Markdown.parse("""
```julia
$(code)
```
""")
````

