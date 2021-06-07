# Define a bunch of configs for testing purposes
struct MostBoringConfig <: RuleConfig{Union{}} end

struct MockForwardsConfig <: RuleConfig{Union{CanForwardsMode, NoReverseMode}}
    forward_calls::Vector
end
MockForwardsConfig() = MockForwardsConfig([])
function ChainRulesCore.frule_via_ad(config::MockForwardsConfig, ārgs, f, args...; kws...)
    # For testing purposes we only support giving right answer for identity functions
    push!(config.forward_calls, (f, args))
    return f(args...; kws...), ārgs
end

struct MockReverseConfig <: RuleConfig{Union{NoForwardsMode, CanReverseMode}}
    reverse_calls::Vector
end
MockReverseConfig() = MockReverseConfig([])
function ChainRulesCore.rrule_via_ad(config::MockReverseConfig, f, args...; kws...)
    # For testing purposes we only support giving right answer for identity functions
    push!(config.reverse_calls, (f, args))
    pullback_via_ad(ȳ) = NoTangent(), ȳ
    return f(args...; kws...), pullback_via_ad
end


struct MockBothConfig <: RuleConfig{Union{CanForwardsMode, CanReverseMode}}
    forward_calls::Vector
    reverse_calls::Vector
end
MockBothConfig() = MockBothConfig([], [])
function ChainRulesCore.frule_via_ad(config::MockBothConfig, ārgs, f, args...; kws...)
    # For testing purposes we only support giving right answer for identity functions
    push!(config.forward_calls, (f, args))
    return f(args...; kws...), ārgs
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
        MostBoringConfig, MockForwardsConfig, MockReverseConfig, MockBothConfig,
    )
        counting_id_count = Ref(0)
        function counting_id(x)
            counting_id_count[]+=1
            return x
        end
        function ChainRulesCore.rrule(::typeof(counting_id), x)
            counting_id_pullback(x̄) = x̄
            return counting_id(x), counting_id_pullback
        end
        function ChainRulesCore.frule((dself, dx),::typeof(counting_id), x)
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
            config::RuleConfig{>:CanForwardsMode}, (_, df, dx), ::typeof(do_thing_2), f, x
        )
            return frule_via_ad(config, (df, dx), f, x)
        end

        @testset "$Config" for Config in (MostBoringConfig, MockReverseConfig)
            @test nothing === frule(
                Config(), (NoTangent(), NoTangent(), 21.5), do_thing_2, identity, 32.1
            )
        end

        @testset "$Config" for Config in (MockBothConfig, MockForwardsConfig)
            bconfig= Config()
            @test nothing !== frule(
                bconfig, (NoTangent(), NoTangent(), 21.5), do_thing_2, identity, 32.1
            )
            @test bconfig.forward_calls == [(identity, (32.1,))]
        end
    end

    @testset "hitting reverse AD" begin
        do_thing_3(f, x) = f(x)
        function ChainRulesCore.rrule(
            config::RuleConfig{>:CanReverseMode}, ::typeof(do_thing_3), f, x
        )
            return (NoTangent(), rrule_via_ad(config, f, x)...)
        end


        @testset "$Config" for Config in (MostBoringConfig, MockForwardsConfig)
            @test nothing === rrule(Config(), do_thing_3, identity, 32.1)
        end

        @testset "$Config" for Config in (MockBothConfig, MockReverseConfig)
            bconfig= Config()
            @test nothing !== rrule(bconfig, do_thing_3, identity, 32.1)
            @test bconfig.reverse_calls == [(identity, (32.1,))]
        end
    end

    @testset "hitting forwards AD from reverse, if available and reverse if not" begin
        # this is is the complicated case doing something interesting and pseudo-mixed mode
        do_thing_4(f, x) = f(x)
        function ChainRulesCore.rrule(
            config::RuleConfig{>:CanForwardsMode},
            ::typeof(do_thing_4),
            f::Function,
            x::Real,
        )
            ẋ = one(x)
            y, ẏ = frule_via_ad(config, (NoTangent, NoTangent(), ẋ), f, x)
            pullback_via_forwards_ad(ȳ) = NoTangent(), ẏ*ȳ
            return y, pullback_via_forwards_ad
        end
        function ChainRulesCore.rrule(
            config::RuleConfig{>:Union{CanReverseMode, NoForwardsMode}},
            ::typeof(do_thing_4),
            f,
            x
        )
            return (NoTangent(), rrule_via_ad(config, f, x)...)
        end

        @test nothing === rrule(MostBoringConfig(), do_thing_4, identity, 32.1)

        @testset "$Config" for Config in (MockBothConfig, MockForwardsConfig)
            bconfig= Config()
            @test nothing !== rrule(bconfig, do_thing_4, identity, 32.1)
            @test bconfig.forward_calls == [(identity, (32.1,))]
        end

        rconfig= MockReverseConfig()
        @test nothing !== rrule(rconfig, do_thing_4, identity, 32.1)
        @test rconfig.reverse_calls == [(identity, (32.1,))]
    end
end
