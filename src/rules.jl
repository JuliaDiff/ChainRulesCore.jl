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
    return y, ∂y
end
```
"""
macro frule(ast)
    return quote
        $(esc(ast))
        $register_new_rule(frule, $(QuoteNode(ast)))
    end
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
    return quote
        $(esc(ast))
        $register_new_rule(rrule, $(QuoteNode(ast)))
    end
end

const FRULES = Vector{Pair{Method, Expr}}[]
const RRULES = Vector{Pair{Method, Expr}}[]
rule_list(::typeof(rrule)) = RRULES
rule_list(::typeof(frule)) = FRULES

function register_new_rule(rule_kind, ast)
    method = _just_defined_method(rule_kind)
    push!(rule_list(rule_kind), method=>ast)
    trigger_new_rule_hooks(rule_kind, method, ast)
    return nothing
end


"""
    _just_defined_method(f)

Finds the method of `f` that was defined in the current world-age.
Errors if not found.
"""
function _just_defined_method(f)
    @static if VERSION >= v"1.2"
        current_world_age = Base.get_world_counter()
        defined_world = :primary_world
    else
        current_world_age = ccall(:jl_get_world_counter, UInt, ())
        defined_world = :min_world
    end
    
    for m in methods(f)
        getproperty(m, defined_world) == current_world_age && return m
    end
    error("No method of `f` was defined in current world age")
end


NEW_RRULE_HOOKS = Function[]
NEW_FRULE_HOOKS = Function[]
hook_list(::typeof(rrule)) = NEW_RRULE_HOOKS
hook_list(::typeof(frule)) = NEW_FRULE_HOOKS

function trigger_new_rule_hooks(rule_kind, method, ast)
    for hook_fun in hook_list(rule_kind)
        try
            hook_fun(method, ast)
        catch err
            @warn "Error triggering hooks" hook_fun method ast exception=err
        end
    end
end

on_new_rule(hook_fun, rule_kind) = push!(hook_list(rule_kind), hook_fun)
on_new_rrule(hook_fun) = on_new_rule(hook_fun, rrule)
on_new_frule(hook_fun) = on_new_rule(hook_fun, frule)
