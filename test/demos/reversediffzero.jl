"The simplest viable reverse mode a AD, only supports `Float64`"
module ReverseDiffZero
using ChainRulesCore
using Test

#########################################
# Initial rule setup
@scalar_rule x + y (1, 1)
@scalar_rule x - y (1, -1)
##########################
#Define the AD

struct Tracked{F} <: Real
    propagate::F
    primal::Float64
    tape::Vector{Tracked}  # a reference to a shared tape
    partial::Base.RefValue{Float64} # current accumulated sensitivity
end

"An intermediate value, a Branch in Nabla terms."
function Tracked(propagate, primal, tape)
    v = Tracked(propagate, primal, tape, Ref(zero(primal)))
    push!(tape, v)
    return v
end

"Marker for inputs (leaves) that don't need to propagate."
struct NoPropagate end

"An input, a Leaf in Nabla terms. No inputs of its own to propagate to."
function Tracked(primal, tape)
    # don't actually need to put these on the tape, since they don't need to propagate
    return Tracked(NoPropagate(), primal, tape, Ref(zero(primal)))
end

primal(d::Tracked) = d.primal
primal(d) = d

partial(d::Tracked) = d.partial[]
partial(d) = nothing

tape(d::Tracked) = d.tape
tape(d) = nothing

"we have many inputs grab the tape from the first one that is tracked"
get_tape(ds) = something(tape.(ds)...)

"propagate the currently stored partial back to my inputs."
propagate!(d::Tracked) = d.propagate(d.partial[])

"Accumulate the sensitivity, if the value is being tracked."
accum!(d::Tracked, x̄) = d.partial[] += x̄
accum!(d, x̄) = nothing

# needed for `^` to work from having `*` defined
Base.to_power_type(x::Tracked) = x

"What to do when a new rrule is declared"
function define_tracked_overload(sig)
    sig = Base.unwrap_unionall(sig)  # not really handling most UnionAll
    opT, argTs = Iterators.peel(sig.parameters)
    opT isa Type{<:Type} && return  # not handling constructors
    fieldcount(opT) == 0 || return  # not handling functors
    all(argT isa Type && Float64 <: argT for argT in argTs) || return # only handling purely Float64 ops.

    N = length(sig.parameters) - 1  # skip the op
    fdef = quote
        # we use the function call overloading form as it lets us avoid namespacing issues
        # as we can directly interpolate the function type into to the AST.
        function (op::$opT)(tracked_args::Vararg{Union{Tracked, Float64}, $N}; kwargs...)
            args = (op, primal.(tracked_args)...)
            y, y_pullback = rrule(args...; kwargs...)
            the_tape = get_tape(tracked_args)
            y_tracked = Tracked(y, the_tape) do ȳ
                # pull this partial back and propagate it to the input's partial store
                _, ārgs = Iterators.peel(y_pullback(ȳ))
                accum!.(tracked_args, ārgs)
            end
            return y_tracked
        end
    end
    eval(fdef)
end

# !Important!: Attach the define function to the `on_new_rule` hook
on_new_rule(define_tracked_overload, rrule)

"Do a calculus. `f` should have a single output."
function derv(f, args::Vararg; kwargs...)
    the_tape = Vector{Tracked}()
    tracked_inputs = Tracked.(args, Ref(the_tape))
    tracked_output = f(tracked_inputs...; kwargs...)
    @assert tape(tracked_output) === the_tape

    # Now the backward pass
    out = primal(tracked_output)
    ōut = one(out)
    accum!(tracked_output, ōut)
    # By going down the tape backwards we know we will have fully accumulated partials
    # before propagating them onwards
    for op in reverse(the_tape)
        propagate!(op)
    end
    return partial.(tracked_inputs)
end

# End AD definition
################################

# add a rule later also
function ChainRulesCore.rrule(::typeof(*), x::Number, y::Number)
    function times_pullback(ΔΩ)
        # we will use thunks here to show we handle them fine.
        return (NO_FIELDS,  @thunk(ΔΩ * y'), @thunk(x' * ΔΩ))
    end
    return x * y, times_pullback
end

# Manual refresh needed as new rule added in same file as AD after the `on_new_rule` call
refresh_rules();

@testset "ReversedDiffZero" begin
    foo(x) = x + x
    @test derv(foo, 1.6) == (2.0,)

    bar(x) = x + 2.1 * x
    @test derv(bar, 1.2) == (3.1,)

    baz(x) = 2.0 * x^2 + 3.0*x + 1.2
    @test derv(baz, 1.7) == (2 * 2.0 * 1.7 + 3.0,)

    qux(x) = foo(x) + bar(x) + baz(x)
    @test derv(qux, 1.7) == ((2 * 2.0 * 1.7 + 3.0) + 3.1 + 2,)

    function quux(x)
        y = 2.0*x + 3.0*x
        return 4.0*y + 5.0*y
    end
    @test derv(quux, 11.1) == (4*(2+3) + 5*(2+3),)
end
end  # module
