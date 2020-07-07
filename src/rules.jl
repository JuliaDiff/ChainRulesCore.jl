#####
##### `frule`/`rrule`
#####

"""
    frule((Δf, Δx...), f, x...)

Expressing the output of `f(x...)` as `Ω`, return the tuple:

    (Ω, ΔΩ)

The second return value is the differential w.r.t. the output.

If no method matching `frule((Δf, Δx...), f, x...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```jldoctest frule
julia> dself = NO_FIELDS;

julia> x = rand()
0.8236475079774124

julia> sinx, Δsinx = frule((dself, 1), sin, x)
(0.7336293678134624, 0.6795498147167869)

julia> sinx == sin(x)
true

julia> Δsinx == cos(x)
true
```

unary input, binary output scalar function:

```jldoctest frule
julia> sincosx, Δsincosx = frule((dself, 1), sincos, x);

julia> sincosx == sincos(x)
true

julia> Δsincosx == (cos(x), -sin(x))
true
```

When defining overloads they should be wrapped with the [`@frule`](@ref) macro.

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

```jldoctest
julia> x = rand();

julia> sinx, sin_pullback = rrule(sin, x);

julia> sinx == sin(x)
true

julia> sin_pullback(1) == (NO_FIELDS, cos(x))
true
```

binary input, unary output scalar function:

```jldoctest
julia> x, y = rand(2);

julia> hypotxy, hypot_pullback = rrule(hypot, x, y);

julia> hypotxy == hypot(x, y)
true

julia> hypot_pullback(1) == (NO_FIELDS, (x / hypot(x, y)), (y / hypot(x, y)))
true
```

When defining overloads they should be wrapped with the [`@rrule`](@ref) macro.

See also: [`frule`](@ref), [`@scalar_rule`](@ref)
"""
rrule(::Any, ::Vararg{Any}; kwargs...) = nothing

"""
    @frule(function ...)

[`frule`](@ref) defining functions should be decorated with this macro.

Example:
```julia
@frule function frule((Δself, Δargs...), ::typeof(foo), args...; kwargs...)
    ...
    return y, ∂Y
end
```
"""
macro frule(expr)
    return esc(expr)
end

"""
    @rrule(function ...)

[`rrule`](@ref) defining functions should be decorated with this macro.

Example:
```julia
@rrule function rrule(::typeof(foo), args...; kwargs...)
    ...
    return y, pullback
end
```
"""
macro rrule(expr)
    return esc(expr)
end
