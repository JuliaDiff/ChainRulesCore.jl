# These are some macros (and supporting functions) to make it easier to define rules.
using Base.Meta

macro strip_linenos(expr)
    return esc(Base.remove_linenums!(expr))
end

"""
    @scalar_rule(f(x₁, x₂, ...),
                 @setup(statement₁, statement₂, ...),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

A convenience macro that generates simple scalar forward or reverse rules using
the provided partial derivatives. Specifically, generates the corresponding
methods for `frule` and `rrule`:

    function ChainRulesCore.frule((NoTangent(), Δx₁, Δx₂, ...), ::typeof(f), x₁::Number, x₂::Number, ...)
        Ω = f(x₁, x₂, ...)
        \$(statement₁, statement₂, ...)
        return Ω, (
                (∂f₁_∂x₁ * Δx₁ + ∂f₁_∂x₂ * Δx₂ + ...),
                (∂f₂_∂x₁ * Δx₁ + ∂f₂_∂x₂ * Δx₂ + ...),
                ...
            )
    end

    function ChainRulesCore.rrule(::typeof(f), x₁::Number, x₂::Number, ...)
        Ω = f(x₁, x₂, ...)
        \$(statement₁, statement₂, ...)
        return Ω, ((ΔΩ₁, ΔΩ₂, ...)) -> (
                NoTangent(),
                ∂f₁_∂x₁ * ΔΩ₁ + ∂f₂_∂x₁ * ΔΩ₂ + ...),
                ∂f₁_∂x₂ * ΔΩ₁ + ∂f₂_∂x₂ * ΔΩ₂ + ...),
                ...
            )
    end

If no type constraints in `f(x₁, x₂, ...)` within the call to `@scalar_rule` are
provided, each parameter in the resulting `frule`/`rrule` definition is given a
type constraint of `Number`.
Constraints may also be explicitly be provided to override the `Number` constraint,
e.g. `f(x₁::Complex, x₂)`, which will constrain `x₁` to `Complex` and `x₂` to
`Number`.

At present this does not support defining for closures/functors.
Thus in reverse-mode, the first returned partial,
representing the derivative with respect to the function itself, is always `NoTangent()`.
And in forward-mode, the first input to the returned propagator is always ignored.

The result of `f(x₁, x₂, ...)` is automatically bound to `Ω`. This
allows the primal result to be conveniently referenced (as `Ω`) within the
derivative/setup expressions.

This macro assumes complex functions are holomorphic. In general, for non-holomorphic
functions, the `frule` and `rrule` must be defined manually.

If the derivative is one, (e.g. for identity functions) `true` can be used as the most 
general multiplicative identity.

The `@setup` argument can be elided if no setup code is need. In other
words:

    @scalar_rule(f(x₁, x₂, ...),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

is equivalent to:

    @scalar_rule(f(x₁, x₂, ...),
                 @setup(nothing),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

For examples, see ChainRules' `rulesets` directory.

See also: [`frule`](@ref), [`rrule`](@ref).
"""
macro scalar_rule(call, maybe_setup, partials...)
    call, setup_stmts, inputs, partials = _normalize_scalarrules_macro_input(
        call, maybe_setup, partials
    )
    f = call.args[1]

    frule_expr = scalar_frule_expr(__source__, f, call, setup_stmts, inputs, partials)
    rrule_expr = scalar_rrule_expr(__source__, f, call, setup_stmts, inputs, partials)

    ############################################################################
    # Final return: building the expression to insert in the place of this macro
    code = quote
        if !($f isa Type) && fieldcount(typeof($f)) > 0
            throw(ArgumentError(
                "@scalar_rule cannot be used on closures/functors (such as $($f))"
            ))
        end

        $(frule_expr)
        $(rrule_expr)
    end
end


"""
    _normalize_scalarrules_macro_input(call, maybe_setup, partials)

returns (in order) the correctly escaped:
    - `call` with out any type constraints
    - `setup_stmts`: the content of `@setup` or `nothing` if that is not provided,
    -  `inputs`: with all args having the constraints removed from call, or
        defaulting to `Number`
    - `partials`: which are all `Expr{:tuple,...}`
"""
function _normalize_scalarrules_macro_input(call, maybe_setup, partials)
    ############################################################################
    # Setup: normalizing input form etc

    if Meta.isexpr(maybe_setup, :macrocall) && maybe_setup.args[1] == Symbol("@setup")
        setup_stmts = map(esc, maybe_setup.args[3:end])
    else
        setup_stmts = (nothing,)
        partials = (maybe_setup, partials...)
    end
    @assert Meta.isexpr(call, :call)

    # Annotate all arguments in the signature as scalars
    inputs = esc.(_constrain_and_name.(call.args[2:end], :Number))
    # Remove annotations and escape names for the call
    call.args[2:end] .= _unconstrain.(call.args[2:end])
    call.args = esc.(call.args)

    # For consistency in code that follows we make all partials tuple expressions
    partials = map(partials) do partial
        if Meta.isexpr(partial, :tuple)
            partial
        else
            length(inputs) == 1 || error("Invalid use of `@scalar_rule`")
            Expr(:tuple, partial)
        end
    end

    return call, setup_stmts, inputs, partials
end


function scalar_frule_expr(__source__, f, call, setup_stmts, inputs, partials)
    n_outputs = length(partials)
    n_inputs = length(inputs)

    # Δs is the input to the propagator rule
    # because this is push-forward there is one per input to the function
    Δs = _propagator_inputs(n_inputs)
    pushforward_returns = map(1:n_outputs) do output_i
        ∂s = partials[output_i].args
        propagation_expr(Δs, ∂s)
    end
    if n_outputs > 1
        # For forward-mode we return a Tangent if output actually a tuple.
        pushforward_returns = Expr(
            :call, :(Tangent{typeof($(esc(:Ω)))}), pushforward_returns...
        )
    else
        pushforward_returns = first(pushforward_returns)
    end

    return @strip_linenos quote
        # _ is the input derivative w.r.t. function internals. since we do not
        # allow closures/functors with @scalar_rule, it is always ignored
        function ChainRulesCore.frule((_, $(Δs...)), ::typeof($f), $(inputs...))
            $(__source__)
            $(esc(:Ω)) = $call
            $(setup_stmts...)
            return $(esc(:Ω)), $pushforward_returns
        end
    end
end

function scalar_rrule_expr(__source__, f, call, setup_stmts, inputs, partials)
    n_outputs = length(partials)
    n_inputs = length(inputs)

    # Δs is the input to the propagator rule
    # because this is a pull-back there is one per output of function
    Δs = _propagator_inputs(n_outputs)

    # 1 partial derivative per input
    pullback_returns = map(1:n_inputs) do input_i
        ∂s = [partial.args[input_i] for partial in partials]
        propagation_expr(Δs, ∂s, true)
    end

    # Multi-output functions have pullbacks with a tuple input that will be destructured
    pullback_input = n_outputs == 1 ? first(Δs) : Expr(:tuple, Δs...)
    pullback = @strip_linenos quote
        @inline function $(esc(propagator_name(f, :pullback)))($pullback_input)
            $(__source__)
            return (NoTangent(), $(pullback_returns...))
        end
    end

    return @strip_linenos quote
        function ChainRulesCore.rrule(::typeof($f), $(inputs...))
            $(__source__)
            $(esc(:Ω)) = $call
            $(setup_stmts...)
            return $(esc(:Ω)), $pullback
        end
    end
end

# For context on why this is important, see
# https://github.com/JuliaDiff/ChainRulesCore.jl/pull/276
"Declares properly hygenic inputs for propagation expressions"
_propagator_inputs(n) = [esc(gensym(Symbol(:Δ, i))) for i in 1:n]

"""
    propagation_expr(Δs, ∂s, _conj = false)

    Returns the expression for the propagation of
    the input gradient `Δs` though the partials `∂s`.
    Specify `_conj = true` to conjugate the partials.
"""
function propagation_expr(Δs, ∂s, _conj = false)
    # This is basically Δs ⋅ ∂s
    _∂s = map(∂s) do ∂s_i
        if _conj
            :(conj($(esc(∂s_i))))
        else
            esc(∂s_i)
        end
    end
    n∂s = length(_∂s)

    summed_∂_mul_Δs = if n∂s > 1
        # Explicit multiplication is only performed for the first pair
        # of partial and gradient.
        init_expr = :((*).($(_∂s[1]), $(Δs[1])))

        # Apply `muladd` iteratively.
        foldl(Iterators.drop(zip(_∂s, Δs), 1); init=init_expr) do ex, (∂s_i, Δs_i)
            :((muladd).($∂s_i, $Δs_i, $ex))
        end
    else
        # Note: we don't want to do broadcasting with only 1 multiply (no `+`),
        # because some arrays overload multiply with scalar. Avoiding
        # broadcasting saves compilation time.
        :($(_∂s[1]) * $(Δs[1]))
    end

    return summed_∂_mul_Δs
end

"""
    propagator_name(f, propname)

Determines a reasonable name for the propagator function.
The name doesn't really matter too much as it is a local function to be returned
by `frule` or `rrule`, but a good name make debugging easier.
`f` should be some form of AST representation of the actual function,
`propname` should be either `:pullback` or `:pushforward`

This is able to deal with fairly complex expressions for `f`:

    julia> propagator_name(:bar, :pushforward)
    :bar_pushforward

    julia> propagator_name(esc(:(Base.Random.foo)), :pullback)
    :foo_pullback
"""
propagator_name(f::Expr, propname::Symbol) = propagator_name(f.args[end], propname)
propagator_name(fname::Symbol, propname::Symbol) = Symbol(fname, :_, propname)
propagator_name(fname::QuoteNode, propname::Symbol) = propagator_name(fname.value, propname)

"""
    @non_differentiable(signature_expression)

A helper to make it easier to declare that a method is not not differentiable.
This is a short-hand for defining an [`frule`](@ref) and [`rrule`](@ref) that
return [`NoTangent()`](@ref) for all partials (even for the function `s̄elf`-partial
itself)

Keyword arguments should not be included.

```jldoctest
julia> @non_differentiable Base.:(==)(a, b)

julia> _, pullback = rrule(==, 2.0, 3.0);

julia> pullback(1.0)
(NoTangent(), NoTangent(), NoTangent())
```

You can place type-constraints in the signature:
```jldoctest
julia> @non_differentiable Base.length(xs::Union{Number, Array})

julia> frule((ZeroTangent(), 1), length, [2.0, 3.0])
(2, NoTangent())
```

!!! warning
    This helper macro covers only the simple common cases.
    It does not support `where`-clauses.
    For these you can declare the `rrule` and `frule` directly

"""
macro non_differentiable(sig_expr)
    Meta.isexpr(sig_expr, :call) || error("Invalid use of `@non_differentiable`")
    has_vararg = _isvararg(sig_expr.args[end])

    primal_name, orig_args = Iterators.peel(sig_expr.args)

    primal_name_sig, primal_name = _split_primal_name(primal_name)
    constrained_args = _constrain_and_name.(orig_args, :Any)
    primal_sig_parts = [primal_name_sig, constrained_args...]

    unconstrained_args = _unconstrain.(constrained_args)

    primal_invoke = if !has_vararg
        :($(primal_name)($(unconstrained_args...)))
    else
        normal_args = unconstrained_args[1:end-1]
        var_arg = unconstrained_args[end]
        :($(primal_name)($(normal_args...), $(var_arg)...))
    end

    quote
        $(_nondiff_frule_expr(__source__, primal_sig_parts, primal_invoke))
        $(_nondiff_rrule_expr(__source__, primal_sig_parts, primal_invoke))
    end
end

"changes `f(x,y)` into `f(x,y; kwargs....)`"
function _with_kwargs_expr(call_expr::Expr, kwargs)
    @assert isexpr(call_expr, :call)
    return Expr(
        :call, call_expr.args[1], Expr(:parameters, :($(kwargs)...)), call_expr.args[2:end]...
    )
end

function _nondiff_frule_expr(__source__, primal_sig_parts, primal_invoke)
    @gensym kwargs
    return @strip_linenos quote
        # Manually defined kw version to save compiler work. See explanation in rules.jl
        function (::Core.kwftype(typeof(ChainRulesCore.frule)))(@nospecialize($kwargs::Any),
                frule::typeof(ChainRulesCore.frule), @nospecialize(::Any), $(map(esc, primal_sig_parts)...))
            return ($(esc(_with_kwargs_expr(primal_invoke, kwargs))), NoTangent())
        end
        function ChainRulesCore.frule(@nospecialize(::Any), $(map(esc, primal_sig_parts)...))
            $(__source__)
            # Julia functions always only have 1 output, so return a single NoTangent()
            return ($(esc(primal_invoke)), NoTangent())
        end
    end
end

function tuple_expression(primal_sig_parts)
    has_vararg = _isvararg(primal_sig_parts[end])
    return if !has_vararg
        num_primal_inputs = length(primal_sig_parts)
        Expr(:tuple, ntuple(_ -> NoTangent(), num_primal_inputs)...)
    else
        num_primal_inputs = length(primal_sig_parts) - 1 # - vararg
        length_expr = :($num_primal_inputs + length($(esc(_unconstrain(primal_sig_parts[end])))))
        @strip_linenos :(ntuple(i -> NoTangent(), $length_expr))
    end
end

function _nondiff_rrule_expr(__source__, primal_sig_parts, primal_invoke)
    esc_primal_sig_parts = map(esc, primal_sig_parts)
    tup_expr = tuple_expression(primal_sig_parts)
    primal_name = first(primal_invoke.args)
    pullback_expr = @strip_linenos quote
        function $(esc(propagator_name(primal_name, :pullback)))(@nospecialize(_))
            return $(tup_expr)
        end
    end

    @gensym kwargs
    return @strip_linenos quote
        # Manually defined kw version to save compiler work. See explanation in rules.jl
        function (::Core.kwftype(typeof(rrule)))($(esc(kwargs))::Any, ::typeof(rrule), $(esc_primal_sig_parts...))
            return ($(esc(_with_kwargs_expr(primal_invoke, kwargs))), $pullback_expr)
        end
        function ChainRulesCore.rrule($(esc_primal_sig_parts...))
            $(__source__)
            return ($(esc(primal_invoke)), $pullback_expr)
        end
    end
end


###########
# Helpers

"""
    _isvararg(expr)

returns true if the expression could represent a vararg

```jldoctest
julia> ChainRulesCore._isvararg(:(x...))
true

julia> ChainRulesCore._isvararg(:(x::Int...))
true

julia> ChainRulesCore._isvararg(:(::Int...))
true

julia> ChainRulesCore._isvararg(:(x::Vararg))
true

julia> ChainRulesCore._isvararg(:(x::Vararg{Int}))
true

julia> ChainRulesCore._isvararg(:(::Vararg))
true

julia> ChainRulesCore._isvararg(:(::Vararg{Int}))
true

julia> ChainRulesCore._isvararg(:(x))
false
````
"""
_isvararg(expr) = false
function _isvararg(expr::Expr)
    Meta.isexpr(expr, :...) && return true
    if Meta.isexpr(expr, :(::))
        constraint = last(expr.args)
        constraint == :Vararg && return true
        Meta.isexpr(constraint, :curly) && first(constraint.args) == :Vararg && return true
    end
    return false
end

"""
splits the first arg of the `call` expression into an expression to use in the signature
and one to use for calling that function
"""
function _split_primal_name(primal_name)
    # e.g. f(x, y)
    if primal_name isa Symbol || Meta.isexpr(primal_name, :(.)) ||
        Meta.isexpr(primal_name, :curly)

        primal_name_sig = :(::$Core.Typeof($primal_name))
        return primal_name_sig, primal_name
    # e.g. (::T)(x, y)
    elseif Meta.isexpr(primal_name, :(::))
        _primal_name = gensym(Symbol(:instance_, primal_name.args[end]))
        primal_name_sig = Expr(:(::), _primal_name, primal_name.args[end])
        return primal_name_sig, _primal_name
    else
        error("invalid primal name: `$primal_name`")
    end
end

"turn both `a` and `a::S` into `a`"
_unconstrain(arg::Symbol) = arg
function _unconstrain(arg::Expr)
    Meta.isexpr(arg, :(::), 2) && return arg.args[1]  # drop constraint.
    Meta.isexpr(arg, :(...), 1) && return _unconstrain(arg.args[1])
    error("malformed arguments: $arg")
end

"turn both `a` and `::constraint` into `a::constraint` etc"
function _constrain_and_name(arg::Expr, _)
    Meta.isexpr(arg, :(::), 2) && return arg  # it is already fine.
    Meta.isexpr(arg, :(::), 1) && return Expr(:(::), gensym(), arg.args[1]) # add name
    Meta.isexpr(arg, :(...), 1) && return Expr(:(...), _constrain_and_name(arg.args[1], :Any))
    error("malformed arguments: $arg")
end
_constrain_and_name(name::Symbol, constraint) = Expr(:(::), name, constraint)  # add type
