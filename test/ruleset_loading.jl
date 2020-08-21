@testset "ruleset_loading.jl" begin
    @testset "on_new_rule" begin
        frule_history = []
        rrule_history = []
        on_new_rule(frule) do sig
            op = sig.parameters[1]
            push!(frule_history, op)
        end
        on_new_rule(rrule) do sig
            op = sig.parameters[1]
            push!(rrule_history, op)
        end

        # Now define some rules
        @scalar_rule x + y (1, 1)
        @scalar_rule x - y (1, -1)
        refresh_rules()

        @test Set(frule_history[end-1:end]) == Set((typeof(+), typeof(-)))
        @test Set(rrule_history[end-1:end]) == Set((typeof(+), typeof(-)))
        
        clear_new_rule_hooks!(frule)
        clear_new_rule_hooks!(rrule)
    end

    @testset "_primal_sig" begin
        _primal_sig = ChainRulesCore._primal_sig
        @testset "frule" begin
            @test isequal(  # DataType without shared type but with constraint
                _primal_sig(frule, Tuple{typeof(frule), Any, typeof(*), Int, Vector{Int}}),
                Tuple{typeof(*), Int, Vector{Int}}
            )
            @test isequal(  # UnionAall without shared type but with constraint
                _primal_sig(frule, Tuple{typeof(frule), Any, typeof(*),  T, Int} where T<:Real),
                Tuple{typeof(*), T, Int} where T<:Real
            )
            @test isequal(  # UnionAall with share type
                _primal_sig(frule, Tuple{typeof(frule), Any, typeof(*), T, Vector{T}} where T),
                Tuple{typeof(*), T, Vector{T}} where T
            )
        end

        @testset "rrule" begin
            @test isequal(  # DataType without shared type but with constraint
                _primal_sig(rrule, Tuple{typeof(rrule), typeof(*), Int, Vector{Int}}),
                Tuple{typeof(*), Int, Vector{Int}}
            )
            @test isequal(  # UnionAall without shared type but with constraint
                _primal_sig(rrule, Tuple{typeof(rrule), typeof(*),  T, Int} where T<:Real),
                Tuple{typeof(*), T, Int} where T<:Real
            )
            @test isequal(  # UnionAall with share type
                _primal_sig(rrule, Tuple{typeof(rrule), typeof(*), T, Vector{T}} where T),
                Tuple{typeof(*), T, Vector{T}} where T
            )
        end
    end
end
