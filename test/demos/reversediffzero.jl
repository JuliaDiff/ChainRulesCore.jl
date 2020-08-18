"The simplest viable reverse mode a AD, only supports `Float64`"
module ReverseDiffZero

using ChainRulesCore
using Test

struct Tracked <: Real
    propagate::Function
    primal::Float64
    tape::Vector{Any}  # a reference to a shared tape
    grad::Base.RefValue{Float64} # current accumulated gradient
end

"An intermediate value, a Branch in Nabla terms."
function Tracked(propagate, primal, tape)
    v = Tracked(propagate, primal, tape, Ref(zero(primal)))
    push!(tape, v)
    return v
end

"An input, a Leaf in Nabla terms. No inputs of its on to propagate to."
function Tracked(primal, tape)
    # don't actually need to put these on the tape, since they don't need to
    # propagate anything
    return Tracked(_->nothing, primal, tape, Ref(zero(primal)))
end

primal(d::Tracked) = d.primal
primal(d) = d

grad(d::Tracked) = d.grad[]
grad(d) = nothing

tape(d::Tracked) = d.tape
tape(d) = nothing

# we have many inputs grab the tape from the first one that is tracked
get_tape(ds) = something(tape.(ds)...)

# propagate the currently stored gradient back to my inputs.
propagate!(d) = nothing
propagate!(d::Tracked) = d.propagate(d.grad[])

# Accumulate gradient, if the value is being tracked.
accum!(d, x̄) = nothing
accum!(d::Tracked, x̄) = d.grad[] += x̄

function define_tracked_overload(sig)
    opT, argTs = Iterators.peel(sig.parameters)
    fieldcount(opT) == 0 || return  # not handling functors 
    all(Float64 <: argT for argT in argTs) || return  # only handling purely Float64 ops.

    N = length(sig.parameters) - 1  # skip the op
    fdef = quote
        # we use the function call overloading form as it lets us avoid namespacing issues
        # as we can directly interpolate the function type into to the AST.
        function (op::$opT)(tracked_args::Vararg{Union{Tracked, Float64}, $N}; kwargs...)
            args = (op, primal.(tracked_args)...)
            res = rrule(args...; kwargs...)
            res === nothing && error("Apparently no rule for $($sig)), but we really thought there was, args=($args)")
            y, y_pullback = res
            t = get_tape(tracked_args)
            y_tracked = Tracked(y, t) do ȳ
                # pull this gradient back and propagate it to the inputs gradient stores
                _, ārgs = Iterators.peel(y_pullback(ȳ))
                accum!.(tracked_args, ārgs)
            end
            return y_tracked
        end
    end
    eval(fdef)
end

function derv(f, args::Vararg; kwargs...)
    the_tape = Vector{Any}()
    tracked_inputs = Tracked.(args, Ref(the_tape))
    tracked_output = f(tracked_inputs...; kwargs...)
    @assert tape(tracked_output) === the_tape

    out = primal(tracked_output)
    function back(ōut)
        accum!(tracked_output, ōut)
        for op in reverse(the_tape)
            # by going down the tape backwards we know we will
            # have accumulated its gradient fully
            propagate!(op)
        end
        return grad.(tracked_inputs)
    end
    return back(one(out))
end

# needed for ^ to work from having `*` defined
Base.to_power_type(x::Tracked) = x

#########################################
# Initial rule setup
@scalar_rule x + y (One(), One())
@scalar_rule x - y (One(), -1)

on_new_rule(define_tracked_overload, rrule)

# add a rule later also
function ChainRulesCore.rrule(::typeof(*), x::Number, y::Number)
    function times_pullback(ΔΩ)
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
    @test derv(baz, 1.7) == (2*2.0*1.7 + 3.0,)

    qux(x) = foo(x) + bar(x) + baz(x)
    @test derv(qux, 1.7) == ((2*2.0*1.7 + 3.0) + 3.1 + 2,)

    function quux(x)
        y = 2.0*x + 3.0*x
        return 4.0*y + 5.0*y
    end
    @test derv(quux, 11.1) == (4*(2+3) + 5*(2+3),)
end
end  # module