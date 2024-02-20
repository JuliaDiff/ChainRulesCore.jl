# For testing Tangent
struct Foo
    x
    y::Float64
end

mutable struct MFoo
    x::Float64
    y
end

# For testing Primal + Tangent performance
struct Bar
    x::Float64
end

# For testing Tangent: it is an invarient of the type that x2 = 2x
# so simple addition can not be defined
struct StructWithInvariant
    x
    x2

    StructWithInvariant(x) = new(x, 2x)
end
@testset "StructuralTangent" begin
    @testset "Tangent" begin
        @testset "empty types" begin
            @test typeof(Tangent{Tuple{}}()) == Tangent{Tuple{},Tuple{}}
        end

        @testset "constructor" begin
            t = (1.0, 2.0)
            nt = (x=1, y=2.0)
            d = Dict(:x => 1.0, :y => 2.0)
            vals = [1, 2]

            @test_throws ArgumentError Tangent{typeof(t),typeof(nt)}(nt)
            @test_throws ArgumentError Tangent{typeof(t),typeof(d)}(d)

            @test_throws ArgumentError Tangent{typeof(d),typeof(nt)}(nt)
            @test_throws ArgumentError Tangent{typeof(d),typeof(t)}(t)

            @test_throws ArgumentError Tangent{typeof(nt),typeof(vals)}(vals)
            @test_throws ArgumentError Tangent{typeof(nt),typeof(d)}(d)
            @test_throws ArgumentError Tangent{typeof(nt),typeof(t)}(t)

            @test_throws ArgumentError Tangent{Foo,typeof(d)}(d)
            @test_throws ArgumentError Tangent{Foo,typeof(t)}(t)
        end

        @testset "==" begin
            @test Tangent{Foo}(; x=0.1, y=2.5) == Tangent{Foo}(; x=0.1, y=2.5)
            @test Tangent{Foo}(; x=0.1, y=2.5) == Tangent{Foo}(; y=2.5, x=0.1)
            @test Tangent{Foo}(; y=2.5, x=ZeroTangent()) == Tangent{Foo}(; y=2.5)

            @test Tangent{Tuple{Float64}}(2.0) == Tangent{Tuple{Float64}}(2.0)
            @test Tangent{Dict}(Dict(4 => 3)) == Tangent{Dict}(Dict(4 => 3))

            tup = (1.0, 2.0)
            @test Tangent{typeof(tup)}(1.0, 2.0) == Tangent{typeof(tup)}(1.0, @thunk(2 * 1.0))
            @test Tangent{typeof(tup)}(1.0, 2.0) == Tangent{typeof(tup)}(1.0, 2)

            @test Tangent{Foo}(; y=2.0) == Tangent{Foo}(; x=ZeroTangent(), y=Float32(2.0))
        end

        @testset "hash" begin
            @test hash(Tangent{Foo}(; x=0.1, y=2.5)) == hash(Tangent{Foo}(; y=2.5, x=0.1))
            @test hash(Tangent{Foo}(; y=2.5, x=ZeroTangent())) == hash(Tangent{Foo}(; y=2.5))
        end

        @testset "indexing, iterating, and properties" begin
            @test keys(Tangent{Foo}(; x=2.5)) == (:x,)
            @test propertynames(Tangent{Foo}(; x=2.5)) == (:x,)
            @test haskey(Tangent{Foo}(; x=2.5), :x) == true
            if isdefined(Base, :hasproperty)
                @test hasproperty(Tangent{Foo}(; x=2.5), :y) == false
            end
            @test Tangent{Foo}(; x=2.5).x == 2.5

            tang1 = Tangent{Tuple{Float64}}(2.0)
            @test keys(tang1) == Base.OneTo(1)
            @test propertynames(Tangent{Tuple{Float64}}(2.0)) == (1,)
            @test getindex(Tangent{Tuple{Float64}}(2.0), 1) == 2.0
            @test getindex(Tangent{Tuple{Float64}}(@thunk 2.0^2), 1) == 4.0
            @test getproperty(Tangent{Tuple{Float64}}(2.0), 1) == 2.0
            @test getproperty(Tangent{Tuple{Float64}}(@thunk 2.0^2), 1) == 4.0
            @test NoTangent() === @inferred Base.tail(tang1)
            @test NoTangent() === @inferred Base.tail(Tangent{Tuple{}}())
            
            tang3 = Tangent{Tuple{Float64, String, Vector{Float64}}}(1.0, NoTangent(), @thunk [3.0] .+ 4)
            @test @inferred(first(tang3)) === tang3[1] === 1.0
            @test @inferred(last(tang3)) isa Thunk
            @test unthunk(last(tang3)) == [7.0]
            @test Tuple(@inferred Base.tail(tang3))[1] === NoTangent()
            @test Tuple(Base.tail(tang3))[end] isa Thunk

            NT = NamedTuple{(:a, :b),Tuple{Float64,Float64}}
            @test getindex(Tangent{NT}(; a=(@thunk 2.0^2)), :a) == 4.0
            @test getindex(Tangent{NT}(; a=(@thunk 2.0^2)), :b) == ZeroTangent()
            @test getindex(Tangent{NT}(; b=(@thunk 2.0^2)), 1) == ZeroTangent()
            @test getindex(Tangent{NT}(; b=(@thunk 2.0^2)), 2) == 4.0

            @test getproperty(Tangent{NT}(; a=(@thunk 2.0^2)), :a) == 4.0
            @test getproperty(Tangent{NT}(; a=(@thunk 2.0^2)), :b) == ZeroTangent()
            @test getproperty(Tangent{NT}(; b=(@thunk 2.0^2)), 1) == ZeroTangent()
            @test getproperty(Tangent{NT}(; b=(@thunk 2.0^2)), 2) == 4.0
            
            @test first(Tangent{NT}(; a=(@thunk 2.0^2))) isa Thunk
            @test unthunk(first(Tangent{NT}(; a=(@thunk 2.0^2)))) == 4.0
            @test last(Tangent{NT}(; a=(@thunk 2.0^2))) isa ZeroTangent
            
            ntang1 = @inferred Base.tail(Tangent{NT}(; b=(@thunk 2.0^2)))
            @test ntang1 isa Tangent{<:NamedTuple{(:b,)}}
            @test NoTangent() === @inferred Base.tail(ntang1)

            # TODO: uncomment this once https://github.com/JuliaLang/julia/issues/35516
            # if VERSION >= v"1.8-"
            #     @test haskey(Tangent{Tuple{Float64}}(2.0), 1) == true
            # else
            #     @test_broken haskey(Tangent{Tuple{Float64}}(2.0), 1) == true
            # end
            @test_broken hasproperty(Tangent{Tuple{Float64}}(2.0), 2) == false

            @test length(Tangent{Foo}(; x=2.5)) == 1
            @test length(Tangent{Tuple{Float64}}(2.0)) == 1

            @test eltype(Tangent{Foo}(; x=2.5)) == Float64
            @test eltype(Tangent{Tuple{Float64}}(2.0)) == Float64

            # Testing iterate via collect
            @test collect(Tangent{Foo}(; x=2.5)) == [2.5]
            @test collect(Tangent{Tuple{Float64}}(2.0)) == [2.0]

            # Test indexed_iterate
            ctup = Tangent{Tuple{Float64,Int64}}(2.0, 3)
            _unpack2tuple = function (tangent)
                a, b = tangent
                return (a, b)
            end
            @inferred _unpack2tuple(ctup)
            @test _unpack2tuple(ctup) === (2.0, 3)

            # Test getproperty is inferrable
            _unpacknamedtuple = tangent -> (tangent.x, tangent.y)
            if VERSION ≥ v"1.2"
                @inferred _unpacknamedtuple(Tangent{Foo}(; x=2, y=3.0))
                @inferred _unpacknamedtuple(Tangent{Foo}(; y=3.0))
            end
        end

        @testset "reverse" begin
            c = Tangent{Tuple{Int,Int,String}}(1, 2, "something")
            cr = Tangent{Tuple{String,Int,Int}}("something", 2, 1)
            @test reverse(c) === cr

            if VERSION < v"1.9-"
                # can't reverse a named tuple or a dict
                @test_throws MethodError reverse(Tangent{Foo}(; x=1.0, y=2.0))

                d = Dict(:x => 1, :y => 2.0)
                cdict = Tangent{typeof(d),typeof(d)}(d)
                @test_throws MethodError reverse(Tangent{Foo}())
            else
                # These now work but do we care?
            end
        end

        @testset "unset properties" begin
            @test Tangent{Foo}(; x=1.4).y === ZeroTangent()
        end

        @testset "conj" begin
            @test conj(Tangent{Foo}(; x=2.0 + 3.0im)) == Tangent{Foo}(; x=2.0 - 3.0im)
            @test ==(
                conj(Tangent{Tuple{Float64}}(2.0 + 3.0im)), Tangent{Tuple{Float64}}(2.0 - 3.0im)
            )
            @test ==(
                conj(Tangent{Dict}(Dict(4 => 2.0 + 3.0im))),
                Tangent{Dict}(Dict(4 => 2.0 + -3.0im)),
            )
        end

        @testset "canonicalize" begin
            # Testing iterate via collect
            @test ==(canonicalize(Tangent{Tuple{Float64}}(2.0)), Tangent{Tuple{Float64}}(2.0))

            @test ==(canonicalize(Tangent{Dict}(Dict(4 => 3))), Tangent{Dict}(Dict(4 => 3)))

            # For structure it needs to match order and ZeroTangent() fill to match primal
            CFoo = Tangent{Foo}
            @test canonicalize(CFoo(; x=2.5, y=10)) == CFoo(; x=2.5, y=10)
            @test canonicalize(CFoo(; y=10, x=2.5)) == CFoo(; x=2.5, y=10)
            @test canonicalize(CFoo(; y=10)) == CFoo(; x=ZeroTangent(), y=10)

            @test_throws ArgumentError canonicalize(CFoo(; q=99.0, x=2.5))

            @testset "unspecified primal type" begin
                c1 = Tangent{Any}(; a=1, b=2)
                c2 = Tangent{Any}(1, 2)
                c3 = Tangent{Any}(Dict(4 => 3))

                @test c1 == canonicalize(c1)
                @test c2 == canonicalize(c2)
                @test c3 == canonicalize(c3)
            end
        end

        @testset "+ with other composites" begin
            @testset "Structs" begin
                CFoo = Tangent{Foo}
                @test CFoo(; x=1.5) + CFoo(; x=2.5) == CFoo(; x=4.0)
                @test CFoo(; y=1.5) + CFoo(; x=2.5) == CFoo(; y=1.5, x=2.5)
                @test CFoo(; y=1.5, x=1.5) + CFoo(; x=2.5) == CFoo(; y=1.5, x=4.0)
            end

            @testset "Tuples" begin
                @test ==(
                    typeof(Tangent{Tuple{}}() + Tangent{Tuple{}}()), Tangent{Tuple{},Tuple{}}
                )
                @test (
                    Tangent{Tuple{Float64,Float64}}(1.0, 2.0) +
                    Tangent{Tuple{Float64,Float64}}(1.0, 1.0)
                ) == Tangent{Tuple{Float64,Float64}}(2.0, 3.0)
            end

            @testset "NamedTuples" begin
                make_tangent(nt::NamedTuple) = Tangent{typeof(nt)}(; nt...)
                t1 = make_tangent((; a=1.5, b=0.0))
                t2 = make_tangent((; a=0.0, b=2.5))
                t_sum = make_tangent((a=1.5, b=2.5))
                @test t1 + t2 == t_sum
            end

            @testset "Dicts" begin
                d1 = Tangent{Dict}(Dict(4 => 3.0, 3 => 2.0))
                d2 = Tangent{Dict}(Dict(4 => 3.0, 2 => 2.0))
                d_sum = Tangent{Dict}(Dict(4 => 3.0 + 3.0, 3 => 2.0, 2 => 2.0))
                @test d1 + d2 == d_sum
            end

            @testset "Fields of type NotImplemented" begin
                CFoo = Tangent{Foo}
                a = CFoo(; x=1.5)
                b = CFoo(; x=@not_implemented(""))
                for (x, y) in ((a, b), (b, a), (b, b))
                    z = x + y
                    @test z isa CFoo
                    @test z.x isa ChainRulesCore.NotImplemented
                end

                a = Tangent{Tuple}(1.5)
                b = Tangent{Tuple}(@not_implemented(""))
                for (x, y) in ((a, b), (b, a), (b, b))
                    z = x + y
                    @test z isa Tangent{Tuple}
                    @test first(z) isa ChainRulesCore.NotImplemented
                end

                a = Tangent{NamedTuple{(:x,)}}(; x=1.5)
                b = Tangent{NamedTuple{(:x,)}}(; x=@not_implemented(""))
                for (x, y) in ((a, b), (b, a), (b, b))
                    z = x + y
                    @test z isa Tangent{NamedTuple{(:x,)}}
                    @test z.x isa ChainRulesCore.NotImplemented
                end

                a = Tangent{Dict}(Dict(:x => 1.5))
                b = Tangent{Dict}(Dict(:x => @not_implemented("")))
                for (x, y) in ((a, b), (b, a), (b, b))
                    z = x + y
                    @test z isa Tangent{Dict}
                    @test z[:x] isa ChainRulesCore.NotImplemented
                end
            end
        end

        @testset "+ with Primals" begin
            @testset "Structs" begin
                @test Foo(3.5, 1.5) + Tangent{Foo}(; x=2.5) == Foo(6.0, 1.5)
                @test Tangent{Foo}(; x=2.5) + Foo(3.5, 1.5) == Foo(6.0, 1.5)
                @test (@ballocated Bar(0.5) + Tangent{Bar}(; x=0.5)) == 0
            end

            @testset "Tuples" begin
                @test Tangent{Tuple{}}() + () == ()
                @test ((1.0, 2.0) + Tangent{Tuple{Float64,Float64}}(1.0, 1.0)) == (2.0, 3.0)
                @test (Tangent{Tuple{Float64,Float64}}(1.0, 1.0)) + (1.0, 2.0) == (2.0, 3.0)
            end

            @testset "NamedTuple" begin
                ntx = (; a=1.5)
                @test Tangent{typeof(ntx)}(; ntx...) + ntx == (; a=3.0)

                nty = (; a=1.5, b=0.5)
                @test Tangent{typeof(nty)}(; nty...) + nty == (; a=3.0, b=1.0)
            end

            @testset "Dicts" begin
                d_primal = Dict(4 => 3.0, 3 => 2.0)
                d_tangent = Tangent{typeof(d_primal)}(Dict(4 => 5.0))
                @test d_primal + d_tangent == Dict(4 => 3.0 + 5.0, 3 => 2.0)
            end
        end

        @testset "+ with Primals, with inner constructor" begin
            value = StructWithInvariant(10.0)
            diff = Tangent{StructWithInvariant}(; x=2.0, x2=6.0)

            @testset "with and without debug mode" begin
                @assert ChainRulesCore.debug_mode() == false
                @test_throws MethodError (value + diff)
                @test_throws MethodError (diff + value)

                ChainRulesCore.debug_mode() = true  # enable debug mode
                @test_throws ChainRulesCore.PrimalAdditionFailedException (value + diff)
                @test_throws ChainRulesCore.PrimalAdditionFailedException (diff + value)
                ChainRulesCore.debug_mode() = false  # disable it again
            end

            # Now we define constuction for ChainRulesCore.jl's purposes:
            # It is going to determine the root quanity of the invarient
            function ChainRulesCore.construct(::Type{StructWithInvariant}, nt::NamedTuple)
                x = (nt.x + nt.x2 / 2) / 2
                return StructWithInvariant(x)
            end
            @test value + diff == StructWithInvariant(12.5)
            @test diff + value == StructWithInvariant(12.5)
        end

        @testset "differential arithmetic" begin
            c = Tangent{Foo}(; y=1.5, x=2.5)

            @test NoTangent() * c == NoTangent()
            @test c * NoTangent() == NoTangent()
            @test dot(NoTangent(), c) == NoTangent()
            @test dot(c, NoTangent()) == NoTangent()
            @test norm(Tangent{Foo}(; y=c.y, x=NoTangent())) == c.y
            @test norm(NoTangent(), Inf) == 0

            @test ZeroTangent() * c == ZeroTangent()
            @test c * ZeroTangent() == ZeroTangent()
            @test dot(ZeroTangent(), c) == ZeroTangent()
            @test dot(c, ZeroTangent()) == ZeroTangent()
            @test norm(ZeroTangent()) == 0
            @test norm(ZeroTangent(), 0.4) == 0

            @test true * c === c
            @test c * true === c

            t = @thunk 2
            @test t * c == 2 * c
            @test c * t == c * 2
        end

        @testset "-Tangent" begin
            t = Tangent{Foo}(; x=1.0, y=-2.0)
            @test -t == Tangent{Foo}(; x=-1.0, y=2.0)
            @test -1.0 * t == -t
        end

        @testset "subtraction" begin
            a = Tangent{Foo}(; x=2.0, y=-2.0)
            b = Tangent{Foo}(; x=1.0, y=2.0)
            @test (a - b) == Tangent{Foo}(; x=1.0, y=-4.0)

            a = Tangent{Foo}(; x=2.0, y=-2.0)
            b = Tangent{Foo}(; x=1.0)
            @test (a - b) == Tangent{Foo}(; x=1.0, y=-2.0)

            a = Tangent{Tuple{Float64,Float64}}(2.0, 3.0)
            b = Tangent{Tuple{Float64,Float64}}(1.0, 1.0)
            @test (a - b) == Tangent{Tuple{Float64,Float64}}(1.0, 2.0)

            a = MutableTangent{MFoo}(; x=1.5, y=1.5)
            b = MutableTangent{MFoo}(; x=0.5, y=0.5)
            @test (a - b) == MutableTangent{MFoo}(; x=1.0, y=1.0)
        end

        @testset "scaling" begin
            @test (
                2 * Tangent{Foo}(; y=1.5, x=2.5) ==
                Tangent{Foo}(; y=3.0, x=5.0) ==
                Tangent{Foo}(; y=1.5, x=2.5) * 2
            )
            @test (
                2 * Tangent{Tuple{Float64,Float64}}(2.0, 4.0) ==
                Tangent{Tuple{Float64,Float64}}(4.0, 8.0) ==
                Tangent{Tuple{Float64,Float64}}(2.0, 4.0) * 2
            )
            d = Tangent{Dict}(Dict(4 => 3.0))
            two_d = Tangent{Dict}(Dict(4 => 2 * 3.0))
            @test 2 * d == two_d == d * 2

            @test_throws MethodError [1, 2] * Tangent{Foo}(; y=1.5, x=2.5)
            @test_throws MethodError [1, 2] * d
            @test_throws MethodError Tangent{Foo}(; y=1.5, x=2.5) * @thunk [1 2; 3 4]
        end

        @testset "scaling division" begin
            a = Tangent{Foo}(; x=2.0, y=-2.0)
            @test a / 2.0 == Tangent{Foo}(; x=1.0, y=-1.0) == 2.0 \ a
            @test (
                Tangent{Tuple{Float64,Float64}}(2.0, 4.0) / 2.0 ==
                Tangent{Tuple{Float64,Float64}}(1.0, 2.0) ==
                2.0 \ Tangent{Tuple{Float64,Float64}}(2.0, 4.0)
            )
        end

        @testset "iszero" begin
            @test iszero(Tangent{Foo}())
            @test iszero(Tangent{Tuple{}}())
            @test iszero(Tangent{Foo}(; x=ZeroTangent()))
            @test iszero(Tangent{Foo}(; y=0.0))
            @test iszero(Tangent{Foo}(; x=Tangent{Tuple{}}(), y=0.0))

            @test !iszero(Tangent{Foo}(; y=3.0))
        end

        @testset "show" begin
            @test repr(Tangent{Foo}(; x=1)) == "Tangent{Foo}(x = 1,)"
            # check for exact regex match not occurence( `^...$`)
            # and allowing optional whitespace (`\s?`)
            @test occursin(
                r"^Tangent{Tuple{Int64,\s?Int64}}\(1,\s?2\)$",
                repr(Tangent{Tuple{Int64,Int64}}(1, 2)),
            )

            @test repr(Tangent{Foo}()) == "Tangent{Foo}()"

            @test ==(
                repr(MutableTangent{MFoo}((; x=1.5, y=[1.0, 2.0]))),
                "MutableTangent{MFoo}(x = 1.5, y = [1.0, 2.0])",
            )
        end

        @testset "internals" begin
            @testset "Can't do backing on primative type" begin
                @test_throws Exception ChainRulesCore.backing(1.4)
            end

            @testset "Internals don't allocate a ton" begin
                bk = (; x=1.0, y=2.0)
                VERSION >= v"1.5" &&
                    @test (@ballocated(ChainRulesCore.construct($Foo, $bk))) <= 32

                # weaker version of the above (which should pass on all versions)
                @test (@ballocated(ChainRulesCore.construct($Foo, $bk))) <= 48
                @test (@ballocated ChainRulesCore.elementwise_add($bk, $bk)) <= 48
            end
        end

        @testset "non-same-typed differential arithmetic" begin
            nt = (; a=1, b=2.0)
            c = Tangent{typeof(nt)}(; a=NoTangent(), b=0.1)
            @test nt + c == (; a=1, b=2.1)
        end
        
        @testset "printing" begin
            t5 = Tuple(rand(3))
            nt3 = (x=t5, y=t5, z=nothing)
            tang = ProjectTo(nt3)(nt3)  # moderately complicated Tangent
            @test contains(sprint(show, tang), "...}(x = Tangent")  # gets shortened
            @test contains(sprint(show, tang), sprint(show, tang.x))  # inner piece appears whole
        end
    end

    @testset "MutableTangent" begin
        mutable struct MDemo
            x::Float64
        end
        function ChainRulesCore.frule(
            (_, ȯbj, _, ẋ), ::typeof(setfield!), obj::MDemo, field, x
        )
            y = setfield!(obj, field, x)
            ẏ = setproperty!(ȯbj, field, ẋ)
            return y, ẏ
        end

        @testset "usecase" begin
            obj = MDemo(99.0)
            ∂obj = MutableTangent{MDemo}(; x=1.5)
            frule((NoTangent(), ∂obj, NoTangent(), 10.0), setfield!, obj, :x, 95.0)
            @test ∂obj.x == 10.0
            @test obj.x == 95.0

            frule((NoTangent(), ∂obj, NoTangent(), 20.0), setfield!, obj, 1, 96.0)
            @test ∂obj.x == 20.0
            @test getproperty(∂obj, 1) == 20.0
            @test obj.x == 96.0
        end

        @testset "== and hash" begin
            @test MutableTangent{MDemo}(; x=1.0f0) == MutableTangent{MDemo}(; x=1.0)
            @test MutableTangent{MDemo}(; x=1.0) == MutableTangent{MDemo}(; x=1.0f0)
            @test MutableTangent{MDemo}(; x=2.0) != MutableTangent{MDemo}(; x=1.0)
            @test MutableTangent{MDemo}(; x=1.0) != MutableTangent{MDemo}(; x=2.0)

            nt = (; x=1.0)
            @test MutableTangent{typeof(nt)}(nt) != MutableTangent{MDemo}(; x=1.0)

            @test hash(MutableTangent{MDemo}(; x=1.0f0)) == hash(MutableTangent{MDemo}(; x=1.0))
        end

        @testset "Mutation" begin
            v = MutableTangent{MFoo}(; x=1.5, y=2.4)
            v.x = 1.6
            @test v == MutableTangent{MFoo}(; x=1.6, y=2.4)
            v.y = [1.0, 2.0]  # change type, because primal can change type
            @test v == MutableTangent{MFoo}(; x=1.6, y=[1.0, 2.0])
        end
    end

    @testset "map" begin
        @testset "Tangent" begin
            ∂foo = Tangent{Foo}(; x=1.5, y=2.4)
            @test map(v -> 2 * v, ∂foo) == Tangent{Foo}(; x=3.0, y=4.8)

            ∂foo = Tangent{Foo}(; x=1.5)
            @test map(v -> 2 * v, ∂foo) == Tangent{Foo}(; x=3.0)
        end
        @testset "MutableTangent" begin
            ∂foo = MutableTangent{MFoo}(; x=1.5, y=2.4)
            ∂foo2 = map(v -> 2 * v, ∂foo)
            @test ∂foo2 == MutableTangent{MFoo}(; x=3.0, y=4.8)
            # Check can still be mutated to new typ
            ∂foo2.y = [1.0, 2.0]
            @test ∂foo2 == MutableTangent{MFoo}(; x=3.0, y=[1.0, 2.0])
        end
    end
end