#####
##### `frule`/`rrule`
#####

# TODO: remember to update the examples
"""
    frule(f, x..., ṡelf, Δx...)

Expressing `x` as the tuple `(x₁, x₂, ...)`, `Δx` as the tuple `(Δx₁, Δx₂,
...)`, and the output tuple of `f(x...)` as `Ω`, return the tuple:

    (Ω, (Ω̇₁, Ω̇₂, ...))

The second return value is the propagation rule, or the pushforward.
It takes in differentials corresponding to the inputs (`ẋ₁, ẋ₂, ...`)
and `ṡelf` the internal values of the function (for closures).


If no method matching `frule(f, x..., ṡelf, Δx...)` has been defined, then
return `nothing`.

Examples:

unary input, unary output scalar function:

```
julia> dself = Zero()
Zero()

julia> x = rand();

julia> sinx, sin_pushforward = frule(sin, x, dself, 1)
(0.35696518021277485, 0.9341176907197836)

julia> sinx == sin(x)
true

julia> sin_pushforward(dself, 1) == cos(x)
true
```

unary input, binary output scalar function:

```
julia> x = rand();

julia> sincosx, sincos_pushforward = frule(sincos, x, dself, 1);

julia> sincosx == sincos(x)
true

julia> sincos_pushforward(dself, 1) == (cos(x), -sin(x))
true
```

See also: [`rrule`](@ref), [`@scalar_rule`](@ref)
"""
frule(::Any, ::Vararg{Any}; kwargs...) = nothing

"""
    rrule(f, x...)

Expressing `x` as the tuple `(x₁, x₂, ...)` and the output tuple of `f(x...)`
as `Ω`, return the tuple:

    (Ω, (Ω̄₁, Ω̄₂, ...) -> (s̄elf, x̄₁, x̄₂, ...))

Where the second return value is the the propagation rule or pullback.
It takes in differentials corresponding to the outputs (`x̄₁, x̄₂, ...`),
and `s̄elf`, the internal values of the function itself (for closures)

If no method matching `rrule(f, xs...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```
julia> x = rand();

julia> sinx, sin_pullback = rrule(sin, x);

julia> sinx == sin(x)
true

julia> sin_pullback(1) == (NO_FIELDS, cos(x))
true
```

binary input, unary output scalar function:

```
julia> x, y = rand(2);

julia> hypotxy, hypot_pullback = rrule(hypot, x, y);

julia> hypotxy == hypot(x, y)
true

julia> hypot_pullback(1) == (NO_FIELDS, (x / hypot(x, y)), (y / hypot(x, y)))
true
```

See also: [`frule`](@ref), [`@scalar_rule`](@ref)
"""
rrule(::Any, ::Vararg{Any}; kwargs...) = nothing
