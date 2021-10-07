# Define a bunch of configs for testing purposes
struct MostBoringConfig <: RuleConfig{Union{}} end

struct MockForwardsConfig <: RuleConfig{Union{HasForwardsMode,NoReverseMode}}
    forward_calls::Vector
end
MockForwardsConfig() = MockForwardsConfig([])
function ChainRulesCore.frule_via_ad(config::MockForwardsConfig, ȧrgs, f, args...; kws...)
    # For testing purposes we only support giving right answer for identity functions
    push!(config.forward_calls, (f, args))
    return f(args...; kws...), ȧrgs
end

struct MockReverseConfig <: RuleConfig{Union{NoForwardsMode,HasReverseMode}}
    reverse_calls::Vector
end
MockReverseConfig() = MockReverseConfig([])
function ChainRulesCore.rrule_via_ad(config::MockReverseConfig, f, args...; kws...)
    # For testing purposes we only support giving right answer for identity functions
    push!(config.reverse_calls, (f, args))
    pullback_via_ad(ȳ) = NoTangent(), ȳ
    return f(args...; kws...), pullback_via_ad
end

struct MockBothConfig <: RuleConfig{Union{HasForwardsMode,HasReverseMode}}
    forward_calls::Vector
    reverse_calls::Vector
end
MockBothConfig() = MockBothConfig([], [])
function ChainRulesCore.frule_via_ad(config::MockBothConfig, ȧrgs, f, args...; kws...)
    # For testing purposes we only support giving right answer for identity functions
    push!(config.forward_calls, (f, args))
    return f(args...; kws...), ȧrgs
end

function ChainRulesCore.rrule_via_ad(config::MockBothConfig, f, args...; kws...)
    # For testing purposes we only support giving right answer for identity functions
    push!(config.reverse_calls, (f, args))
    pullback_via_ad(ȳ) = NoTangent(), ȳ
    return f(args...; kws...), pullback_via_ad
end

##############################

#define some functions for testing

@testset "config.jl" begin
    @testset "basic fall to two arg verion for $Config" for Config in (
        MostBoringConfig, MockForwardsConfig, MockReverseConfig, MockBothConfig
    )
        counting_id_count = Ref(0)
        function counting_id(x)
            counting_id_count[] += 1
            return x
        end
        function ChainRulesCore.rrule(::typeof(counting_id), x)
            counting_id_pullback(x̄) = x̄
            return counting_id(x), counting_id_pullback
        end
        function ChainRulesCore.frule((dself, dx), ::typeof(counting_id), x)
            return counting_id(x), dx
        end
        @testset "rrule" begin
            counting_id_count[] = 0
            @test rrule(Config(), counting_id, 21.5) !== nothing
            @test counting_id_count[] == 1
        end
        @testset "frule" begin
            counting_id_count[] = 0
            @test frule(Config(), (NoTangent(), 11.2), counting_id, 32.4) !== nothing
            @test counting_id_count[] == 1
        end
    end

    @testset "hitting forwards AD" begin
        do_thing_2(f, x) = f(x)
        function ChainRulesCore.frule(
            config::RuleConfig{>:HasForwardsMode}, (_, df, dx), ::typeof(do_thing_2), f, x
        )
            return frule_via_ad(config, (df, dx), f, x)
        end

        @testset "$Config" for Config in (MostBoringConfig, MockReverseConfig)
            @test nothing === frule(
                Config(), (NoTangent(), NoTangent(), 21.5), do_thing_2, identity, 32.1
            )
        end

        @testset "$Config" for Config in (MockBothConfig, MockForwardsConfig)
            bconfig = Config()
            @test nothing !== frule(
                bconfig, (NoTangent(), NoTangent(), 21.5), do_thing_2, identity, 32.1
            )
            @test bconfig.forward_calls == [(identity, (32.1,))]
        end
    end

    @testset "hitting reverse AD" begin
        do_thing_3(f, x) = f(x)
        function ChainRulesCore.rrule(
            config::RuleConfig{>:HasReverseMode}, ::typeof(do_thing_3), f, x
        )
            return (NoTangent(), rrule_via_ad(config, f, x)...)
        end

        @testset "$Config" for Config in (MostBoringConfig, MockForwardsConfig)
            @test nothing === rrule(Config(), do_thing_3, identity, 32.1)
        end

        @testset "$Config" for Config in (MockBothConfig, MockReverseConfig)
            bconfig = Config()
            @test nothing !== rrule(bconfig, do_thing_3, identity, 32.1)
            @test bconfig.reverse_calls == [(identity, (32.1,))]
        end
    end

    @testset "hitting forwards AD from reverse, if available and reverse if not" begin
        # this is is the complicated case doing something interesting and pseudo-mixed mode
        do_thing_4(f, x) = f(x)
        function ChainRulesCore.rrule(
            config::RuleConfig{>:HasForwardsMode},
            ::typeof(do_thing_4),
            f::Function,
            x::Real,
        )
            # real code would support functors/closures, but in interest of keeping example short we exclude it:
            @assert (fieldcount(typeof(f)) == 0) "Functors/Closures are not supported"

            ẋ = one(x)
            y, ẏ = frule_via_ad(config, (NoTangent(), ẋ), f, x)
            pullback_via_forwards_ad(ȳ) = NoTangent(), NoTangent(), ẏ * ȳ
            return y, pullback_via_forwards_ad
        end
        function ChainRulesCore.rrule(
            config::RuleConfig{>:Union{HasReverseMode,NoForwardsMode}},
            ::typeof(do_thing_4),
            f,
            x,
        )
            y, f_pullback = rrule_via_ad(config, f, x)
            do_thing_4_pullback(ȳ) = (NoTangent(), f_pullback(ȳ)...)
            return y, do_thing_4_pullback
        end

        @test nothing === rrule(MostBoringConfig(), do_thing_4, identity, 32.1)

        @testset "$Config" for Config in (MockBothConfig, MockForwardsConfig)
            bconfig = Config()
            @test nothing !== rrule(bconfig, do_thing_4, identity, 32.1)
            @test bconfig.forward_calls == [(identity, (32.1,))]
        end

        rconfig = MockReverseConfig()
        @test nothing !== rrule(rconfig, do_thing_4, identity, 32.1)
        @test rconfig.reverse_calls == [(identity, (32.1,))]
    end

    @testset "RuleConfig broadcasts like a scaler" begin
        @test (MostBoringConfig() .=> (1, 2, 3)) isa NTuple{3,Pair{MostBoringConfig,Int}}
    end

    @testset "fallbacks" begin
        no_rule(x; kw="bye") = error()
        @test frule((1.0,), no_rule, 2.0) === nothing
        @test frule((1.0,), no_rule, 2.0; kw="hello") === nothing
        @test frule(MostBoringConfig(), (1.0,), no_rule, 2.0) === nothing
        @test frule(MostBoringConfig(), (1.0,), no_rule, 2.0; kw="hello") === nothing
        @test rrule(no_rule, 2.0) === nothing
        @test rrule(no_rule, 2.0; kw="hello") === nothing
        @test rrule(MostBoringConfig(), no_rule, 2.0) === nothing
        @test rrule(MostBoringConfig(), no_rule, 2.0; kw="hello") === nothing

        # Test that incorrect use of the fallback rules correctly throws MethodError
        @test_throws MethodError frule()
        @test_throws MethodError frule(; kw="hello")
        @test_throws MethodError frule(sin)
        @test_throws MethodError frule(sin; kw="hello")
        @test_throws MethodError frule(MostBoringConfig())
        @test_throws MethodError frule(MostBoringConfig(); kw="hello")
        @test_throws MethodError frule(MostBoringConfig(), sin)
        @test_throws MethodError frule(MostBoringConfig(), sin; kw="hello")
        @test_throws MethodError rrule()
        @test_throws MethodError rrule(; kw="hello")
        @test_throws MethodError rrule(MostBoringConfig())
        @test_throws MethodError rrule(MostBoringConfig(); kw="hello")
    end
end
