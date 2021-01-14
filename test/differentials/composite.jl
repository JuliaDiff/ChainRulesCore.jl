# For testing Composite
struct Foo
    x
    y::Float64
end

# For testing Primal + Composite performance
struct Bar
    x::Float64
end

# For testing Composite: it is an invarient of the type that x2 = 2x
# so simple addition can not be defined
struct StructWithInvariant
    x
    x2

    StructWithInvariant(x) = new(x, 2x)
end

@testset "Composite" begin
    @testset "empty types" begin
        @test typeof(Composite{Tuple{}}()) == Composite{Tuple{}, Tuple{}}
    end
    @testset "convert" begin
        @test convert(NamedTuple, Composite{Foo}(x=2.5)) == (; x=2.5)
        @test convert(Tuple, Composite{Tuple{Float64,}}(2.0)) == (2.0,)
        @test convert(Dict, Composite{Dict}(Dict(4 => 3))) == Dict(4 => 3)
    end

    @testset "==" begin
        @test Composite{Foo}(x=0.1, y=2.5) == Composite{Foo}(x=0.1, y=2.5)
        @test Composite{Foo}(x=0.1, y=2.5) == Composite{Foo}(y=2.5, x=0.1)
        @test Composite{Foo}(y=2.5, x=Zero()) == Composite{Foo}(y=2.5)

        @test Composite{Tuple{Float64,}}(2.0) == Composite{Tuple{Float64,}}(2.0)
        @test Composite{Dict}(Dict(4 => 3)) == Composite{Dict}(Dict(4 => 3))

        tup = (1.0, 2.0)
        @test Composite{typeof(tup)}(1.0, 2.0) == Composite{typeof(tup)}(1.0, @thunk(2*1.0))
        @test Composite{typeof(tup)}(1.0, 2.0) == Composite{typeof(tup)}(1.0, 2)

        @test Composite{Foo}(;y=2.0,) == Composite{Foo}(;x=Zero(), y=Float32(2.0),)
    end

    @testset "hash" begin
        @test hash(Composite{Foo}(x=0.1, y=2.5)) == hash(Composite{Foo}(y=2.5, x=0.1))
        @test hash(Composite{Foo}(y=2.5, x=Zero())) == hash(Composite{Foo}(y=2.5))
    end

    @testset "indexing, iterating, and properties" begin
        @test keys(Composite{Foo}(x=2.5)) == (:x,)
        @test propertynames(Composite{Foo}(x=2.5)) == (:x,)
        @test haskey(Composite{Foo}(x=2.5), :x) == true
        if isdefined(Base, :hasproperty)
            @test hasproperty(Composite{Foo}(x=2.5), :y) == false
        end
        @test Composite{Foo}(x=2.5).x == 2.5

        @test keys(Composite{Tuple{Float64,}}(2.0)) == Base.OneTo(1)
        @test propertynames(Composite{Tuple{Float64,}}(2.0)) == (1,)
        @test getproperty(Composite{Tuple{Float64,}}(2.0), 1) == 2.0
        @test getproperty(Composite{Tuple{Float64,}}(@thunk 2.0^2), 1) == 4.0
        @test getproperty(Composite{Tuple{Float64,}}(a=(@thunk 2.0^2),), :a) == 4.0

        # TODO: uncomment this once https://github.com/JuliaLang/julia/issues/35516
        @test_broken haskey(Composite{Tuple{Float64}}(2.0), 1) == true
        @test_broken hasproperty(Composite{Tuple{Float64}}(2.0), 2) == false

        @test length(Composite{Foo}(x=2.5)) == 1
        @test length(Composite{Tuple{Float64,}}(2.0)) == 1

        @test eltype(Composite{Foo}(x=2.5)) == Float64
        @test eltype(Composite{Tuple{Float64,}}(2.0)) == Float64

        # Testing iterate via collect
        @test collect(Composite{Foo}(x=2.5)) == [2.5]
        @test collect(Composite{Tuple{Float64,}}(2.0)) == [2.0]

        # Test indexed_iterate
        ctup = Composite{Tuple{Float64,Int64}}(2.0, 3)
        _unpack2tuple = function(comp)
            a, b = comp
            return (a, b)
        end
        @inferred _unpack2tuple(ctup)
        @test _unpack2tuple(ctup) === (2.0, 3)

        # Test getproperty is inferrable
        _unpacknamedtuple = comp -> (comp.x, comp.y)
        if VERSION ≥ v"1.2"
            @inferred _unpacknamedtuple(Composite{Foo}(x=2, y=3.0))
            @inferred _unpacknamedtuple(Composite{Foo}(y=3.0))
        end
    end

    @testset "reverse" begin
        c = Composite{Tuple{Int, Int, String}}(1, 2, "something")
        cr = Composite{Tuple{String, Int, Int}}("something", 2, 1)
        @test reverse(c) === cr

        # can't reverse a named tuple or a dict
        @test_throws MethodError reverse(Composite{Foo}(;x=1.0, y=2.0))

        d = Dict(:x => 1, :y => 2.0)
        cdict = Composite{Foo, typeof(d)}(d)
        @test_throws MethodError reverse(Composite{Foo}()) 
    end

    @testset "unset properties" begin
        @test Composite{Foo}(; x=1.4).y === Zero()
    end

    @testset "conj" begin
        @test conj(Composite{Foo}(x=2.0+3.0im)) == Composite{Foo}(x=2.0-3.0im)
        @test ==(
            conj(Composite{Tuple{Float64,}}(2.0+3.0im)),
            Composite{Tuple{Float64,}}(2.0-3.0im)
        )
        @test ==(
            conj(Composite{Dict}(Dict(4 => 2.0 + 3.0im))),
            Composite{Dict}(Dict(4 => 2.0 + -3.0im)),
        )
    end

    @testset "extern" begin
        @test extern(Composite{Foo}(x=2.0)) == (;x=2.0)
        @test extern(Composite{Tuple{Float64,}}(2.0)) == (2.0,)
        @test extern(Composite{Dict}(Dict(4 => 3))) == Dict(4 => 3)

        # with differentials on the inside
        @test extern(Composite{Foo}(x=@thunk(0+2.0))) == (;x=2.0)
        @test extern(Composite{Tuple{Float64,}}(@thunk(0+2.0))) == (2.0,)
        @test extern(Composite{Dict}(Dict(4 => @thunk(3)))) == Dict(4 => 3)
    end

    @testset "canonicalize" begin
        # Testing iterate via collect
        @test ==(
            canonicalize(Composite{Tuple{Float64,}}(2.0)),
            Composite{Tuple{Float64,}}(2.0)
        )

        @test ==(
            canonicalize(Composite{Dict}(Dict(4 => 3))),
            Composite{Dict}(Dict(4 => 3)),
        )

        # For structure it needs to match order and Zero() fill to match primal
        CFoo = Composite{Foo}
        @test canonicalize(CFoo(x=2.5, y=10)) == CFoo(x=2.5, y=10)
        @test canonicalize(CFoo(y=10, x=2.5)) == CFoo(x=2.5, y=10)
        @test canonicalize(CFoo(y=10)) == CFoo(x=Zero(), y=10)

        @test_throws ArgumentError canonicalize(CFoo(q=99.0, x=2.5))
    end

    @testset "+ with other composites" begin
        @testset "Structs" begin
            CFoo = Composite{Foo}
            @test  CFoo(x=1.5) + CFoo(x=2.5) == CFoo(x=4.0)
            @test CFoo(y=1.5) + CFoo(x=2.5) == CFoo(y=1.5, x=2.5)
            @test CFoo(y=1.5, x=1.5) + CFoo(x=2.5) == CFoo(y=1.5, x=4.0)
        end

        @testset "Tuples" begin
            @test ==(
                typeof(Composite{Tuple{}}() + Composite{Tuple{}}()),
                Composite{Tuple{}, Tuple{}}
            )
            @test (
                Composite{Tuple{Float64, Float64}}(1.0, 2.0) +
                Composite{Tuple{Float64, Float64}}(1.0, 1.0)
            ) == Composite{Tuple{Float64, Float64}}(2.0, 3.0)
        end

        @testset "NamedTuples" begin
            nt1 = (;a=1.5, b=0.0)
            nt2 = (;a=0.0, b=2.5)
            nt_sum = (a=1.5, b=2.5)
            @test (
                Composite{typeof(nt1)}(; nt1...) +
                Composite{typeof(nt2)}(; nt2...)
            ) == Composite{typeof(nt_sum)}(; nt_sum...)
        end

        @testset "Dicts" begin
            d1 = Composite{Dict}(Dict(4 => 3.0, 3 => 2.0))
            d2 = Composite{Dict}(Dict(4 => 3.0, 2 => 2.0))
            d_sum = Composite{Dict}(Dict(4 => 3.0 + 3.0, 3 => 2.0, 2 => 2.0))
            @test d1 + d2 == d_sum
        end
    end

    @testset "+ with Primals" begin
        @testset "Structs" begin
            @test Foo(3.5, 1.5) + Composite{Foo}(x=2.5) == Foo(6.0, 1.5)
            @test Composite{Foo}(x=2.5) + Foo(3.5, 1.5) == Foo(6.0, 1.5)
            @test (@ballocated Bar(0.5) + Composite{Bar}(; x=0.5)) == 0
        end

        @testset "Tuples" begin
            @test Composite{Tuple{}}() + () == ()
            @test ((1.0, 2.0) + Composite{Tuple{Float64, Float64}}(1.0, 1.0)) == (2.0, 3.0)
            @test (Composite{Tuple{Float64, Float64}}(1.0, 1.0)) + (1.0, 2.0) == (2.0, 3.0)
        end

        @testset "NamedTuple" begin
            ntx = (; a=1.5)
            @test Composite{typeof(ntx)}(; ntx...) + ntx == (; a=3.0)

            nty = (; a=1.5, b=0.5)
            @test Composite{typeof(nty)}(; nty...) + nty == (; a=3.0, b=1.0)
        end

        @testset "Dicts" begin
            d_primal = Dict(4 => 3.0, 3 => 2.0)
            d_tangent = Composite{typeof(d_primal)}(Dict(4 =>5.0))
            @test d_primal + d_tangent == Dict(4 => 3.0 + 5.0, 3 => 2.0)
        end
    end

    @testset "+ with Primals, with inner constructor" begin
        value = StructWithInvariant(10.0)
        diff = Composite{StructWithInvariant}(x=2.0, x2=6.0)

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
            x = (nt.x + nt.x2/2)/2
            return StructWithInvariant(x)
        end
        @test value + diff == StructWithInvariant(12.5)
        @test diff + value == StructWithInvariant(12.5)
    end

    @testset "differential arithmetic" begin
        c = Composite{Foo}(y=1.5, x=2.5)

        @test DoesNotExist() * c == DoesNotExist()
        @test c * DoesNotExist() == DoesNotExist()
        @test dot(DoesNotExist(), c) == DoesNotExist()
        @test dot(c, DoesNotExist()) == DoesNotExist()

        @test Zero() * c == Zero()
        @test c * Zero() == Zero()
        @test dot(Zero(), c) == Zero()
        @test dot(c, Zero()) == Zero()

        @test One() * c === c
        @test c * One() === c

        t = @thunk 2
        @test t * c == 2 * c
        @test c * t == c * 2
    end

    @testset "scaling" begin
        @test (
            2 *  Composite{Foo}(y=1.5, x=2.5)
            == Composite{Foo}(y=3.0, x=5.0)
            == Composite{Foo}(y=1.5, x=2.5) * 2
        )
        @test (
            2 * Composite{Tuple{Float64, Float64}}(2.0, 4.0)
            == Composite{Tuple{Float64, Float64}}(4.0, 8.0)
            == Composite{Tuple{Float64, Float64}}(2.0, 4.0) * 2
        )
        d = Composite{Dict}(Dict(4 => 3.0))
        two_d = Composite{Dict}(Dict(4 => 2 * 3.0))
        @test 2 * d == two_d == d * 2
    end

    @testset "show" begin
        @test repr(Composite{Foo}(x=1,)) == "Composite{Foo}(x = 1,)"
        # check for exact regex match not occurence( `^...$`)
        # and allowing optional whitespace (`\s?`)
        @test occursin(
            r"^Composite{Tuple{Int64,\s?Int64}}\(1,\s?2\)$",
            repr(Composite{Tuple{Int64,Int64}}(1, 2)),
        )
    end

    @testset "internals" begin
        @testset "Can't do backing on primative type" begin
            @test_throws Exception ChainRulesCore.backing(1.4)
        end

        @testset "Internals don't allocate a ton" begin
            bk = (; x=1.0, y=2.0)
            VERSION >= v"1.5" && @test (@ballocated(ChainRulesCore.construct($Foo, $bk))) <= 32
            
            # weaker version of the above (which should pass on all versions)
            @test (@ballocated(ChainRulesCore.construct($Foo, $bk))) <= 48
            @test (@ballocated ChainRulesCore.elementwise_add($bk, $bk)) <= 48
        end
    end

    @testset "non-same-typed differential arithmetic" begin
        nt = (; a=1, b=2.0)
        c = Composite{typeof(nt)}(; a=DoesNotExist(), b=0.1)
        @test nt + c == (; a=1, b=2.1);
    end

    @testset "NO_FIELDS" begin
        @test NO_FIELDS === Zero()
    end
end
