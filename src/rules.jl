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

See also: [`frule`](@ref), [`@scalar_rule`](@ref)
"""
rrule(::Any, ::Vararg{Any}; kwargs...) = nothing

"""
    _rule_list(frule | rrule)

Returns a list of all rules currently defined rules of the given kind.
"""
_rule_list
_rule_list(::typeof(rrule)) = RRULES
_rule_list(::typeof(frule)) = FRULES
const SIGT = Type
const FRULES = Vector{Pair{SIGT, Expr}}()
const RRULES = Vector{Pair{SIGT, Expr}}()


function _register_new_rule(rule_kind, ast)
    method = _just_defined_method(rule_kind)
    rule_sig::SIGT = method.sig 
    sig = _primal_sig(rule_kind, rule_sig)
    push!(_rule_list(rule_kind), sig=>ast)
    _trigger_new_rule_hooks(rule_kind, sig, ast)
    return nothing
end

function _primal_sig(::typeof(frule), rule_sig)
    @assert rule_sig.parameters[1] == typeof(frule)
    # need to skip frule and the deriviative info, so starting from the 3rd
    return Tuple{rule_sig.parameters[3:end]...}
end

function _primal_sig(::typeof(rrule), rule_sig)
    @assert rule_sig.parameters[1] == typeof(rrule)
    # need to skip rrule so starting from the 2rd
    return Tuple{rule_sig.parameters[2:end]...}
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
_hook_list(::typeof(rrule)) = NEW_RRULE_HOOKS
_hook_list(::typeof(frule)) = NEW_FRULE_HOOKS

function _trigger_new_rule_hooks(rule_kind, sig, ast)
    for hook_fun in _hook_list(rule_kind)
        try
            hook_fun(sig, ast)
        catch err
            @warn "Error triggering hooks" hook_fun sig ast exception=err
        end
    end
end

"""
    on_new_rule(((sig, ast)->eval...), frule | rrule)
"""
function on_new_rule(hook_fun, rule_kind)
    # get all the existing rules
    ret = map(_rule_list(rule_kind)) do (sig, ast)
        hook_fun(sig, ast)
    end

    # register hook for new rules
    push!(_hook_list(rule_kind), hook_fun)
    return ret
end
