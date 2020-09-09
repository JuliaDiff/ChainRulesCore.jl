# Infastructure to support generating overloads from rules.
function __init__()
    # Need to refresh rules when a package is loaded
    push!(Base.package_callbacks, pkgid -> refresh_rules())
end

# Holds all the hook functions that are invokes when a new rule is defined
const RRULE_DEFINITION_HOOKS = Function[]
const FRULE_DEFINITION_HOOKS = Function[]
_hook_list(::typeof(rrule)) = RRULE_DEFINITION_HOOKS
_hook_list(::typeof(frule)) = FRULE_DEFINITION_HOOKS

"""
    on_new_rule(hook, frule | rrule)

Register a `hook` function to run when new rules are defined.
The hook receives a signature type-type as input, and generally will use `eval` to define
an overload of an AD system's overloaded type
For example, using the signature type `Tuple{typeof(+), Real, Real}` to make
`+(::DualNumber, ::DualNumber)` call the `frule` for `+`.
A signature type tuple always has the form:
`Tuple{typeof(operation), typeof{pos_arg1}, typeof{pos_arg2}...}`, where `pos_arg1` is the
first positional argument.

The hooks are automatically run on new rules whenever a package is loaded.
They can be manually triggered by [`refresh_rules`](@ref).
When a hook is first registered with `on_new_rule` it is run on all existing rules.
"""
function on_new_rule(hook_fun, rule_kind)
    # apply the hook to the existing rules
    ret = map(_rule_list(rule_kind)) do method
        sig = _primal_sig(rule_kind, method)
        _safe_hook_fun(hook_fun, sig)
    end

    # register hook for new rules -- so all new rules get this function applied
    push!(_hook_list(rule_kind), hook_fun)
    return ret
end

"""
    clear_new_rule_hooks!(frule|rrule)

Clears all hooks that were registered with corresponding [`on_new_rule`](@ref).
This is useful for while working interactively to define your rule generating hooks.
If you previously wrong an incorrect hook, you can use this to get rid of the old one.

!!! warning
    This absolutely should not be used in a package, as it will break any other AD system
    using the rule hooks that might happen to be loaded.
"""
clear_new_rule_hooks!(rule_kind) = empty!(_hook_list(rule_kind))

"""
    _rule_list(frule | rrule)

Returns a list of all the methods of the currently defined rules of the given kind.
Excluding the fallback rule that returns `nothing` for every input.
"""
function _rule_list end
# The fallback rules are the only rules defined in ChainRulesCore & that is how we skip them
_rule_list(rule_kind) = (m for m in methods(rule_kind) if m.module != @__MODULE__)


const LAST_REFRESH_RRULE = Ref(0)
const LAST_REFRESH_FRULE = Ref(0)
last_refresh(::typeof(frule)) = LAST_REFRESH_FRULE
last_refresh(::typeof(rrule)) = LAST_REFRESH_RRULE

"""
    refresh_rules()
    refresh_rules(frule | rrule)

This triggers all [`on_new_rule`](@ref) hooks to run on any newly defined rules.
It is *automatically* run when ever a package is loaded.
It can also be manually called to run it directly, for example if a rule was defined
in the REPL or within the same file as the AD function.
"""
function refresh_rules()
    refresh_rules(frule);
    refresh_rules(rrule)
end

function refresh_rules(rule_kind)
    isempty(_rule_list(rule_kind)) && return  # if no hooks, exit early, nothing to run
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
function _primal_sig(::typeof(frule), rule_sig::DataType)
    @assert rule_sig.parameters[1] == typeof(frule)
    # need to skip frule and the deriviative info, so starting from the 3rd
    return Tuple{rule_sig.parameters[3:end]...}
end
function _primal_sig(::typeof(rrule), rule_sig::DataType)
    @assert rule_sig.parameters[1] == typeof(rrule)
    # need to skip rrule so starting from the 2rd
    return Tuple{rule_sig.parameters[2:end]...}
end
function _primal_sig(rule_kind, rule_sig::UnionAll)
    # This looks a lot like Base.unwrap_unionall and Base.rewrap_unionall, but using those
    # seems not to work
    p_sig = _primal_sig(rule_kind, rule_sig.body)
    return UnionAll(rule_sig.var, p_sig)
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
        @error "Error triggering hook" hook_fun sig exception=(err, catch_backtrace())
    end
end
