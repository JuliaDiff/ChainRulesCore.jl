"The simplest viable forward mode a AD, only supports `Float64`"
module ForwardDiffZero

using ChainRulesCore
using Test

struct Dual <: Real
    primal::Float64
    diff::Float64
end

primal(d::Dual) = d.primal
diff(d::Dual) = d.diff

primal(d::Real) = d
diff(d::Real) = 0.0

# needed for ^ to work from having `*` defined
Base.to_power_type(x::Dual) = x


function define_dual_overload(sig)
    opT, argTs = Iterators.peel(sig.parameters)
    fieldcount(opT) == 0 || return  # not handling functors 
    all(Float64 <: argT for argT in argTs) || return  # only handling purely Float64 ops.

    N = length(sig.parameters) - 1  # skip the op
    fdef = quote
        # we use the function call overloading form as it lets us avoid namespacing issues
        # as we can directly interpolate the function type into to the AST.
        function (op::$opT)(dual_args::Vararg{Union{Dual, Float64}, $N}; kwargs...)
            ȧrgs = (NO_FIELDS,  diff.(dual_args)...)
            args = (op, primal.(dual_args)...)
            res = frule(ȧrgs, args...; kwargs...)
            res === nothing && error("Apparently no rule for $($sig)), but we really thought there was, args=($args)")
            y, ẏ = res
            return Dual(y, ẏ)  # if y, ẏ are not `Float64` this will error.
        end
    end
    #@show fdef
    eval(fdef)
end

#########################################
# Initial rule setup
@scalar_rule x + y (One(), One())
@scalar_rule x - y (One(), -1)

on_new_rule(define_dual_overload, frule)

# add a rule later also
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(*), x::Number, y::Number)
    return (x * y, (Δx * y + x * Δy))
end

# Manual refresh needed as new rule added in same file as AD after the `on_new_rule` call
refresh_rules(); 

function derv(f, arg)
    duals = Dual(arg, one(arg))
    return diff(f(duals...))
end

@testset "ForwardDiffZero" begin
    foo(x) = x + x
    @test derv(foo, 1.6) == 2

    bar(x) = x + 2.1 * x
    @test derv(bar, 1.2) == 3.1

    baz(x) = 2.0 * x^2 + 3.0*x + 1.2
    @test derv(baz, 1.7) == 2*2.0*1.7 + 3.0

    qux(x) = foo(x) + bar(x) + baz(x)
    @test derv(qux, 1.7) == (2*2.0*1.7 + 3.0) + 3.1 + 2

    function quux(x)
        y = 2.0*x + 3.0*x
        return 4.0*y + 5.0*y
    end
    @test derv(quux, 11.1) == 4*(2+3) + 5*(2+3)
end

end  # module