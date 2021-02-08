"The simplest viable forward mode a AD, only supports `Float64`"
module ForwardDiffZero
using ChainRulesCore
using Test

#########################################
# Initial rule setup
@scalar_rule x + y (1, 1)
@scalar_rule x - y (1, -1)
##########################
# Define the AD

# Note that we never directly define Dual Number Arithmetic on Dual numbers
# instead it is automatically defined from the `frules`
struct Dual <: Real
    primal::Float64
    partial::Float64
end

primal(d::Dual) = d.primal
partial(d::Dual) = d.partial

primal(d::Real) = d
partial(d::Real) = 0.0

# needed for `^` to work from having `*` defined
Base.to_power_type(x::Dual) = x


function define_dual_overload(sig)
    sig = Base.unwrap_unionall(sig)  # Not really handling most UnionAlls
    opT, argTs = Iterators.peel(sig.parameters)
    opT isa Type{<:Type} && return  # not handling constructors
    fieldcount(opT) == 0 || return  # not handling functors
    all(argT isa Type && Float64 <: argT for argT in argTs) || return  # only handling purely Float64 ops.

    N = length(sig.parameters) - 1  # skip the op
    fdef = quote
        # we use the function call overloading form as it lets us avoid namespacing issues
        # as we can directly interpolate the function type into to the AST.
        function (op::$opT)(dual_args::Vararg{Union{Dual, Float64}, $N}; kwargs...)
            ȧrgs = (NO_FIELDS,  partial.(dual_args)...)
            args = (op, primal.(dual_args)...)
            y, ẏ = frule(ȧrgs, args...; kwargs...)
            return Dual(y, ẏ)  # if y, ẏ are not `Float64` this will error.
        end
    end
    eval(fdef)
end

# !Important!: Attach the define function to the `on_new_rule` hook
on_new_rule(define_dual_overload, frule)

"Do a calculus. `f` should have a single input."
function derv(f, arg)
    duals = Dual(arg, one(arg))
    return partial(f(duals...))
end

# End AD definition
################################

# add a rule later also
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(*), x::Number, y::Number)
    return (x * y, Δx * y + x * Δy)
end

# Manual refresh needed as new rule added in same file as AD after the `on_new_rule` call
refresh_rules();

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
