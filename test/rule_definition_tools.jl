"""
Along same lines as  `@test_throws` but to test if a macro throw an exception when it is
expanded.
"""
macro test_macro_throws(err_expr, expr)
    quote
        err = nothing
        try
            @macroexpand($(esc(expr)))
        catch load_err
            # all errors thrown at macro expansion time are LoadErrors, we need to unwrap
            @assert load_err isa LoadError
            err = load_err.error
        end
        # Reuse `@test_throws` logic
        if err!==nothing
            @test_throws $(esc(err_expr)) ($(Meta.quot(expr)); throw(err))
        else
            @test_throws $(esc(err_expr)) $(Meta.quot(expr))
        end
    end
end


@testset "rule_definition_tools.jl" begin
    @testset "@non_differentiable" begin
        @testset "two input one output function" begin
            nondiff_2_1(x, y) = fill(7.5, 100)[x + y]
            @non_differentiable nondiff_2_1(::Any, ::Any)
            @test frule((Zero(), 1.2, 2.3), nondiff_2_1, 3, 2) == (7.5, DoesNotExist())
            res, pullback = rrule(nondiff_2_1, 3, 2)
            @test res == 7.5
            @test pullback(4.5) == (NO_FIELDS, DoesNotExist(), DoesNotExist())
        end

        @testset "one input, 2-tuple output function" begin
            nondiff_1_2(x) = (5.0, 3.0)
            @non_differentiable nondiff_1_2(::Any)
            @test frule((Zero(), 1.2), nondiff_1_2, 3.1) == ((5.0, 3.0), DoesNotExist())
            res, pullback = rrule(nondiff_1_2, 3.1)
            @test res == (5.0, 3.0)
            @test isequal(
                pullback(Composite{Tuple{Float64, Float64}}(1.2, 3.2)),
                (NO_FIELDS, DoesNotExist()),
            )
        end

        @testset "constrained signature" begin
            nonembed_identity(x) = x
            @non_differentiable nonembed_identity(::Integer)

            @test frule((Zero(), 1.2), nonembed_identity, 2) == (2, DoesNotExist())
            @test frule((Zero(), 1.2), nonembed_identity, 2.0) == nothing

            res, pullback = rrule(nonembed_identity, 2)
            @test res == 2
            @test pullback(1.2) == (NO_FIELDS, DoesNotExist())

            @test rrule(nonembed_identity, 2.0) == nothing
        end

        @testset "Pointy UnionAll constraints" begin
            pointy_identity(x) = x
            @non_differentiable pointy_identity(::Vector{<:AbstractString})

            @test frule((Zero(), 1.2), pointy_identity, ["2"]) == (["2"], DoesNotExist())
            @test frule((Zero(), 1.2), pointy_identity, 2.0) == nothing

            res, pullback = rrule(pointy_identity, ["2"])
            @test res == ["2"]
            @test pullback(1.2) == (NO_FIELDS, DoesNotExist())

            @test rrule(pointy_identity, 2.0) == nothing
        end

        @testset "Not supported (Yet)" begin
            # Varargs are not supported
            @test_macro_throws ErrorException @non_differentiable vararg1(xs...)
            @test_macro_throws ErrorException @non_differentiable vararg1(xs::Vararg)

            # Where clauses are not supported.
            @test_macro_throws(
                ErrorException,
                (@non_differentiable where_identity(::Vector{T}) where T<:AbstractString)
            )
        end
    end
end
