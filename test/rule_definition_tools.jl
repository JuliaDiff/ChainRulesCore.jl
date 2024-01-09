"""
Along same lines as  `@test_throws` but to test if a macro throw an exception when it is
expanded.
"""
macro test_macro_throws(err_expr, expr)
    quote
        err = nothing
        try
            @macroexpand($(esc(expr)))
        catch _err
            # https://github.com/JuliaLang/julia/pull/38379
            if VERSION >= v"1.7.0-DEV.937"
                err = _err
            else
                # until Julia v1.7
                # all errors thrown at macro expansion time are LoadErrors, we need to unwrap
                @assert _err isa LoadError
                err = _err.error
            end
        end
        # Reuse `@test_throws` logic
        if err !== nothing
            @test_throws $(esc(err_expr)) ($(Meta.quot(expr)); throw(err))
        else
            @test_throws $(esc(err_expr)) $(Meta.quot(expr))
        end
    end
end

# struct need to be defined outside of tests for julia 1.0 compat
struct NonDiffExample
    x
end

struct NonDiffCounterExample
    x
end

module NonDiffModuleExample
nondiff_2_1(x, y) = fill(7.5, 100)[x + y]
end

@testset "rule_definition_tools.jl" begin
    @testset "@non_differentiable" begin
        @testset "two input one output function" begin
            nondiff_2_1(x, y) = fill(7.5, 100)[x + y]
            @non_differentiable nondiff_2_1(::Any, ::Any)
            @test frule((ZeroTangent(), 1.2, 2.3), nondiff_2_1, 3, 2) == (7.5, NoTangent())
            res, pullback = rrule(nondiff_2_1, 3, 2)
            @test res == 7.5
            @test pullback(4.5) == (NoTangent(), NoTangent(), NoTangent())
        end

        @testset "one input, 2-tuple output function" begin
            nondiff_1_2(x) = (5.0, 3.0)
            @non_differentiable nondiff_1_2(::Any)
            @test frule((ZeroTangent(), 1.2), nondiff_1_2, 3.1) == ((5.0, 3.0), NoTangent())
            res, pullback = rrule(nondiff_1_2, 3.1)
            @test res == (5.0, 3.0)
            @test isequal(
                pullback(Tangent{Tuple{Float64,Float64}}(1.2, 3.2)),
                (NoTangent(), NoTangent()),
            )
        end

        @testset "constrained signature" begin
            nonembed_identity(x) = x
            @non_differentiable nonembed_identity(::Integer)

            @test frule((ZeroTangent(), 1.2), nonembed_identity, 2) == (2, NoTangent())
            @test frule((ZeroTangent(), 1.2), nonembed_identity, 2.0) == nothing

            res, pullback = rrule(nonembed_identity, 2)
            @test res == 2
            @test pullback(1.2) == (NoTangent(), NoTangent())

            @test rrule(nonembed_identity, 2.0) == nothing
        end

        @testset "Pointy UnionAll constraints" begin
            pointy_identity(x) = x
            @non_differentiable pointy_identity(::Vector{<:AbstractString})

            @test frule((ZeroTangent(), 1.2), pointy_identity, ["2"]) ==
                  (["2"], NoTangent())
            @test frule((ZeroTangent(), 1.2), pointy_identity, 2.0) == nothing

            res, pullback = rrule(pointy_identity, ["2"])
            @test res == ["2"]
            @test pullback(1.2) == (NoTangent(), NoTangent())

            @test rrule(pointy_identity, 2.0) == nothing
        end

        @testset "kwargs" begin
            kw_demo(x; kw=2.0) = x + kw
            @non_differentiable kw_demo(::Any)

            @testset "not setting kw" begin
                @assert kw_demo(1.5) == 3.5

                res, pullback = rrule(kw_demo, 1.5)
                @test res == 3.5
                @test pullback(4.1) == (NoTangent(), NoTangent())

                @test frule((ZeroTangent(), 11.1), kw_demo, 1.5) == (3.5, NoTangent())
            end

            @testset "setting kw" begin
                @assert kw_demo(1.5; kw=3.0) == 4.5

                res, pullback = rrule(kw_demo, 1.5; kw=3.0)
                @test res == 4.5
                @test pullback(1.1) == (NoTangent(), NoTangent())

                @test frule((ZeroTangent(), 11.1), kw_demo, 1.5; kw=3.0) ==
                      (4.5, NoTangent())
            end
        end

        @testset "Constructors" begin
            @non_differentiable NonDiffExample(::Any)

            @test isequal(
                frule((ZeroTangent(), 1.2), NonDiffExample, 2.0),
                (NonDiffExample(2.0), NoTangent()),
            )

            res, pullback = rrule(NonDiffExample, 2.0)
            @test res == NonDiffExample(2.0)
            @test pullback(1.2) == (NoTangent(), NoTangent())

            # https://github.com/JuliaDiff/ChainRulesCore.jl/issues/213
            # problem was that `@nondiff Foo(x)` was also defining rules for other types.
            # make sure that isn't happenning
            @test frule((ZeroTangent(), 1.2), NonDiffCounterExample, 2.0) === nothing
            @test rrule(NonDiffCounterExample, 2.0) === nothing
        end

        @testset "Varargs" begin
            fvarargs(a, xs...) = sum((a, xs...))
            @testset "xs::Float64..." begin
                @non_differentiable fvarargs(a, xs::Float64...)

                y, pb = rrule(fvarargs, 1)
                @test y == fvarargs(1)
                @test pb(1) == (NoTangent(), NoTangent())

                y, pb = rrule(fvarargs, 1, 2.0, 3.0)
                @test y == fvarargs(1, 2.0, 3.0)
                @test pb(1) == (NoTangent(), NoTangent(), NoTangent(), NoTangent())

                @test frule((1, 1), fvarargs, 1, 2.0) == (fvarargs(1, 2.0), NoTangent())

                @test frule((1, 1), fvarargs, 1, 2) == nothing
                @test rrule(fvarargs, 1, 2) == nothing
            end

            @testset "::Float64..." begin
                @non_differentiable fvarargs(a, ::Float64...)

                y, pb = rrule(fvarargs, 1, 2.0, 3.0)
                @test y == fvarargs(1, 2.0, 3.0)
                @test pb(1) == (NoTangent(), NoTangent(), NoTangent(), NoTangent())

                @test frule((1, 1), fvarargs, 1, 2.0) == (fvarargs(1, 2.0), NoTangent())
            end

            @testset "::Vararg{Float64}" begin
                @non_differentiable fvarargs(a, ::Vararg{Float64})

                y, pb = rrule(fvarargs, 1, 2.0, 3.0)
                @test y == fvarargs(1, 2.0, 3.0)
                @test pb(1) == (NoTangent(), NoTangent(), NoTangent(), NoTangent())

                @test frule((1, 1), fvarargs, 1, 2.0) == (fvarargs(1, 2.0), NoTangent())
            end

            @testset "::Vararg" begin
                @non_differentiable fvarargs(a, ::Vararg)
                @test frule((1, 1), fvarargs, 1, 2) == (fvarargs(1, 2), NoTangent())

                y, pb = rrule(fvarargs, 1, 1)
                @test y == fvarargs(1, 1)
                @test pb(1) == (NoTangent(), NoTangent(), NoTangent())
            end

            @testset "xs..." begin
                @non_differentiable fvarargs(a, xs...)
                @test frule((1, 1), fvarargs, 1, 2) == (fvarargs(1, 2), NoTangent())

                y, pb = rrule(fvarargs, 1, 1)
                @test y == fvarargs(1, 1)
                @test pb(1) == (NoTangent(), NoTangent(), NoTangent())
            end
        end

        @testset "Functors" begin
            (f::NonDiffExample)(y) = fill(7.5, 100)[f.x + y]
            @non_differentiable (::NonDiffExample)(::Any)
            @test frule((Tangent{NonDiffExample}(; x=1.2), 2.3), NonDiffExample(3), 2) ==
                  (7.5, NoTangent())
            res, pullback = rrule(NonDiffExample(3), 2)
            @test res == 7.5
            @test pullback(4.5) == (NoTangent(), NoTangent())
        end

        @testset "Module specified explicitly" begin
            @non_differentiable NonDiffModuleExample.nondiff_2_1(::Any, ::Any)
            @test frule(
                (ZeroTangent(), 1.2, 2.3), NonDiffModuleExample.nondiff_2_1, 3, 2
            ) == (7.5, NoTangent())
            res, pullback = rrule(NonDiffModuleExample.nondiff_2_1, 3, 2)
            @test res == 7.5
            @test pullback(4.5) == (NoTangent(), NoTangent(), NoTangent())
        end

        @testset "interactions with configs" begin
            struct AllConfig <: RuleConfig{Union{HasForwardsMode,NoReverseMode}} end

            foo_ndc1(x) = string(x)
            @non_differentiable foo_ndc1(x)
            @test frule(AllConfig(), (NoTangent(), NoTangent()), foo_ndc1, 2.0) == (string(2.0), NoTangent())
            r1, pb1 = rrule(AllConfig(), foo_ndc1, 2.0)
            @test r1 == string(2.0)
            @test pb1(NoTangent()) == (NoTangent(), NoTangent())

            foo_ndc2(x; y=0) = string(x + y)
            @non_differentiable foo_ndc2(x)
            @test frule(AllConfig(), (NoTangent(), NoTangent()), foo_ndc2, 2.0; y=4.0) == (string(6.0), NoTangent())
            r2, pb2 = rrule(AllConfig(), foo_ndc2, 2.0; y=4.0)
            @test r2 == string(6.0)
            @test pb2(NoTangent()) == (NoTangent(), NoTangent())
        end

        @testset "Not supported (Yet)" begin
            # Where clauses are not supported.
            @test_macro_throws(
                ErrorException,
                (@non_differentiable where_identity(::Vector{T}) where {T<:AbstractString})
            )
        end
    end

    @testset "@scalar_rule" begin
        @testset "@scalar_rule with multiple output" begin
            simo(x) = (x, 2x)
            @scalar_rule(simo(x), 1.0f0, 2.0f0)

            y, simo_pb = rrule(simo, π)

            @test simo_pb((10.0f0, 20.0f0)) == (NoTangent(), 50.0f0)

            y, ẏ = frule((NoTangent(), 50.0f0), simo, π)
            @test y == (π, 2π)
            @test ẏ == Tangent{typeof(y)}(50.0f0, 100.0f0)
            # make sure type is exactly as expected:
            @test ẏ isa Tangent{Tuple{Irrational{:π},Float64},Tuple{Float32,Float32}}

            xs, Ω = (3,), (3, 6)
            @test ChainRulesCore.derivatives_given_output(Ω, simo, xs...) ==
                  ((1.0f0,), (2.0f0,))
        end

        @testset "@scalar_rule projection" begin
            make_imaginary(x) = im * x
            @scalar_rule make_imaginary(x) im

            # note: the === will make sure that these are Float64, not ComplexF64
            @test (NoTangent(), 1.0) === rrule(make_imaginary, 2.0)[2](1.0 * im)
            @test (NoTangent(), 0.0) === rrule(make_imaginary, 2.0)[2](1.0)

            @test (NoTangent(), 1.0 + 0.0im) === rrule(make_imaginary, 2.0im)[2](1.0 * im)
            @test (NoTangent(), 0.0 - 1.0im) === rrule(make_imaginary, 2.0im)[2](1.0)
        end

        @testset "Regression tests against #276 and #265" begin
            # https://github.com/JuliaDiff/ChainRulesCore.jl/pull/276
            # https://github.com/JuliaDiff/ChainRulesCore.jl/pull/265
            # Symptom of these problems is creation of global variables and type instability

            num_globals_before = length(names(ChainRulesCore; all=true))

            simo2(x) = (x, 2x)
            @scalar_rule(simo2(x), 1.0, 2.0)
            _, simo2_pb = rrule(simo2, 43.0)
            # make sure it infers: inferability implies type stability
            @inferred simo2_pb(Tangent{Tuple{Float64,Float64}}(3.0, 6.0))

            # Test no new globals were created
            @test length(names(ChainRulesCore; all=true)) == num_globals_before

            # Example in #265
            simo3(x) = sincos(x)
            @scalar_rule simo3(x) @setup((sinx, cosx) = Ω) cosx -sinx
            _, simo3_pb = @inferred rrule(simo3, randn())
            @inferred simo3_pb(Tangent{Tuple{Float64,Float64}}(randn(), randn()))
        end
    end
end

#! format: off
# workaround for https://github.com/domluna/JuliaFormatter.jl/issues/484
module IsolatedModuleForTestingScoping
    # check that rules can be defined by macros without any additional imports
    using ChainRulesCore: @scalar_rule, @non_differentiable, @opt_out

    # ensure that functions, types etc. in module `ChainRulesCore` can't be resolved
    const ChainRulesCore = nothing

    # this is
    # https://github.com/JuliaDiff/ChainRulesCore.jl/issues/317
    fixed(x) = :abc
    @non_differentiable fixed(x)

    # check name collision between a primal input called `kwargs` and the actual keyword
    # arguments
    fixed_kwargs(x; kwargs...) = :abc
    @non_differentiable fixed_kwargs(kwargs)

    my_id(x) = x
    @scalar_rule(my_id(x), 1.0)

    # @opt_out
    first_oa(x, y) = x
    @scalar_rule(first_oa(x, y), (1, 0))
    # Declared without using the ChainRulesCore namespace qualification
    # see https://github.com/JuliaDiff/ChainRulesCore.jl/issues/545
    @opt_out rrule(::typeof(first_oa), x::T, y::T) where {T<:Float16}
    @opt_out frule(::Any, ::typeof(first_oa), x::T, y::T) where {T<:Float16}

    module IsolatedSubmodule
        # check that rules defined in isolated module without imports can be called
        # without errors
        using ChainRulesCore: frule, rrule, ZeroTangent, NoTangent, derivatives_given_output
        using ChainRulesCore: no_rrule, no_frule
        using ..IsolatedModuleForTestingScoping: fixed, fixed_kwargs, my_id, first_oa
        using Test

        @testset "@non_differentiable" begin
            for f in (fixed, fixed_kwargs)
                y, ẏ = frule((ZeroTangent(), randn()), f, randn())
                @test y === :abc
                @test ẏ === NoTangent()

                y, f_pullback = rrule(f, randn())
                @test y === :abc
                @test f_pullback(randn()) === (NoTangent(), NoTangent())
            end

            y, f_pullback = rrule(fixed_kwargs, randn(); keyword=randn())
            @test y === :abc
            @test f_pullback(randn()) === (NoTangent(), NoTangent())
        end

        @testset "@scalar_rule" begin
            x, ẋ = randn(2)
            y, ẏ = frule((ZeroTangent(), ẋ), my_id, x)
            @test y == x
            @test ẏ == ẋ

            Δy = randn()
            y, f_pullback = rrule(my_id, x)
            @test y == x
            @test f_pullback(Δy) == (NoTangent(), Δy)

            @test derivatives_given_output(y, my_id, x) == ((1.0,),)
        end

        @testset "@optout" begin
            # rrule
            @test rrule(first_oa, Float16(3.0), Float16(4.0)) === nothing
            @test !isempty(
                Iterators.filter(methods(no_rrule)) do m
                    m.sig <: Tuple{Any,typeof(first_oa),T,T} where {T<:Float16}
                end,
            )

            # frule
            @test frule((NoTangent(), 1, 0), first_oa, Float16(3.0), Float16(4.0)) ===
                nothing
            @test !isempty(
                Iterators.filter(methods(no_frule)) do m
                    m.sig <: Tuple{Any,Any,typeof(first_oa),T,T} where {T<:Float16}
                end,
            )
        end
    end
end
#! format: on
