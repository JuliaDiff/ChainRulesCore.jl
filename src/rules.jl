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


#######################################################################
# Infastructure to support generating overloads from rules.

const NEW_RRULE_HOOKS = Function[]
const NEW_FRULE_HOOKS = Function[]
_hook_list(::typeof(rrule)) = NEW_RRULE_HOOKS
_hook_list(::typeof(frule)) = NEW_FRULE_HOOKS

"""
    on_new_rule(sig->eval...), frule | rrule)
"""
function on_new_rule(hook_fun, rule_kind)
    # get all the existing rules
    ret = map(_rule_list(rule_kind)) do method
        sig = _primal_sig(rule_kind, method)
        _safe_hook_fun(hook_fun, sig)
    end

    # register hook for new rules
    push!(_hook_list(rule_kind), hook_fun)
    return ret
end

function __init__()
    push!(Base.package_callbacks, pkgid -> refresh_rules())
    push!(Base.include_callbacks, (mod, filename) -> refresh_rules())
end


"""
    _rule_list(frule | rrule)

Returns a list of all the methods of the currently defined rules of the given kind.
Excluding the fallback rule that returns `nothing` for every input.
"""
_rule_list(rule_kind) = (m for m in methods(rule_kind) if m.module != @__MODULE__)
# ^ The fallback rules are the only rules defined in ChainRules core  so that is how we skip them.



const LAST_REFRESH_RRULE = Ref(0)
const LAST_REFRESH_FRULE = Ref(0)
last_refresh(::typeof(frule)) = LAST_REFRESH_FRULE
last_refresh(::typeof(rrule)) = LAST_REFRESH_RRULE

"""
    refresh_rules()
    refresh_rules(frule | rrule)

This triggers all [`on_new_rule`](@ref) hooks to run on any newly defined rules.
It is *automatically* run when ever a package is loaded, or a file is `include`d.
It can also be manually called to run it directly, for example if a rule was defined
in the REPL or with-in the same file as the AD function.
"""
refresh_rules() = (refresh_rules(frule); refresh_rules(rrule))
function refresh_rules(rule_kind)
    already_done_world_age = last_refresh(rule_kind)[]
    for method in _rule_list(rule_kind)
        _defined_world(method) < already_done_world_age && continue
        sig = _primal_sig(rule_kind, method)
        _trigger_new_rule_hooks(rule_kind, sig)
    end

    last_refresh(rule_kind)[] = _current_world()
    return nothing
end

@static if VERSION >= v"1.2"
    _current_world() = Base.get_world_counter()
    _defined_world(method) = method.primary_world
else
    _current_world() = ccall(:jl_get_world_counter, UInt, ())
    _defined_world(method) = method.min_world
end

"""
    _primal_sig(frule|rule, rule_method | rule_sig)

Returns the signature as a `Tuple{function_type, arg1_type, arg2_type,...}`.
"""
_primal_sig(rule_kind, method::Method) = _primal_sig(rule_kind, method.sig)
function _primal_sig(::typeof(frule), rule_sig::Type)
    @assert rule_sig.parameters[1] == typeof(frule)
    # need to skip frule and the deriviative info, so starting from the 3rd
    return Tuple{rule_sig.parameters[3:end]...}
end

function _primal_sig(::typeof(rrule), rule_sig::Type)
    @assert rule_sig.parameters[1] == typeof(rrule)
    # need to skip rrule so starting from the 2rd
    return Tuple{rule_sig.parameters[2:end]...}
end

function _trigger_new_rule_hooks(rule_kind, sig)
    for hook_fun in _hook_list(rule_kind)
        _safe_hook_fun(hook_fun, sig)
    end
end

function _safe_hook_fun(hook_fun, sig)
    try
        hook_fun(sig)
    catch err
        @error "Error triggering hook" hook_fun sig exception=err
    end
end
