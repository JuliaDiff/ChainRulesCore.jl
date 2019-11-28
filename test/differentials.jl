
# For testing Composite
struct Foo
    x
    y::Float64
end

# For testing Composite: it is an invarient of the type that x2 = 2x
# so simple addition can not be defined
struct StructWithInvariant
    x
    x2

    StructWithInvariant(x) = new(x, 2x)
end


@testset "Differentials" begin
    @testset "Wirtinger" begin
        w = Wirtinger(1+1im, 2+2im)
        @test wirtinger_primal(w) == 1+1im
        @test wirtinger_conjugate(w) == 2+2im
        @test w + w == Wirtinger(2+2im, 4+4im)

        @test w + One() == w + 1 == w + Thunk(()->1) == Wirtinger(2+1im, 2+2im)
        @test w * One() == One() * w == w
        @test w * 2 == 2 * w == Wirtinger(2 + 2im, 4 + 4im)

        # TODO: other + methods stack overflow
        @test_throws ErrorException w*w
        @test_throws ArgumentError extern(w)
        for x in w
            @test x === w
        end
        @test broadcastable(w) == w
        @test_throws MethodError conj(w)
    end
    @testset "Zero" begin
        z = Zero()
        @test extern(z) === false
        @test z + z == z
        @test z + 1 == 1
        @test 1 + z == 1
        @test z * z == z
        @test z * 1 == z
        @test 1 * z == z
        for x in z
            @test x === z
        end
        @test broadcastable(z) isa Ref{Zero}
        @test conj(z) == z
    end
    @testset "One" begin
        o = One()
        @test extern(o) === true
        @test o + o == 2
        @test o + 1 == 2
        @test 1 + o == 2
        @test o * o == o
        @test o * 1 == 1
        @test 1 * o == 1
        for x in o
            @test x === o
        end
        @test broadcastable(o) isa Ref{One}
        @test conj(o) == o
    end

    @testset "Thunk" begin
        @test @thunk(3) isa Thunk

        @testset "show" begin
            rep = repr(Thunk(rand))
            @test occursin(r"Thunk\(.*rand.*\)", rep)
        end

        @testset "Externing" begin
            @test extern(@thunk(3)) == 3
            @test extern(@thunk(@thunk(3))) == 3
        end

        @testset "unthunk" begin
            @test unthunk(@thunk(3)) == 3
            @test unthunk(@thunk(@thunk(3))) isa Thunk
        end

        @testset "calling thunks should call inner function" begin
            @test (@thunk(3))() == 3
            @test (@thunk(@thunk(3)))() isa Thunk
        end

        @testset "erroring thunks should include the source in the backtrack" begin
            expected_line = (@__LINE__) + 2  # for testing it is at right palce
            try
                x = @thunk(error())
                extern(x)
            catch err
                err isa ErrorException || rethrow()
                st = stacktrace(catch_backtrace())
                # Should be 2nd last line, as last line will be the `error` function
                stackframe = st[2]
                @test stackframe.line == expected_line
                @test stackframe.file == Symbol(@__FILE__)
            end
        end
    end

    @testset "Composite" begin
        @testset "convert" begin
            @test convert(NamedTuple, Composite{Foo}(x=2.5)) == (; x=2.5)
            @test convert(Tuple, Composite{Tuple{Float64,}}(2.0)) == (2.0,)
        end

        @testset "indexing, iterating, and properties" begin
            @test propertynames(Composite{Foo}(x=2.5)) == (:x,)
            @test Composite{Foo}(x=2.5).x == 2.5

            @test propertynames(Composite{Tuple{Float64,}}(2.0)) == (1,)
            @test getproperty(Composite{Tuple{Float64,}}(2.0), 1) == 2.0

            @test length(Composite{Foo}(x=2.5)) == 1
            @test length(Composite{Tuple{Float64,}}(2.0)) == 1

            @test eltype(Composite{Foo}(x=2.5)) == Float64
            @test eltype(Composite{Tuple{Float64,}}(2.0)) == Float64

            # Testing iterate via collect
            @test collect(Composite{Foo}(x=2.5)) == [2.5]
            @test collect(Composite{Tuple{Float64,}}(2.0)) == [2.0]
        end

        @testset "conj" begin
            @test conj(Composite{Foo}(x=2.0+3.0im)) == Composite{Foo}(x=2.0-3.0im)
            @test ==(
                conj(Composite{Tuple{Float64,}}(2.0+3.0im)),
                Composite{Tuple{Float64,}}(2.0-3.0im)
            )
        end

        @testset "extern" begin
            @test extern(Composite{Foo}(x=2.0)) == (;x=2.0)
            @test extern(Composite{Tuple{Float64,}}(2.0)) == (2.0,)

            # with differentials on the inside
            @test extern(Composite{Foo}(x=@thunk(0+2.0))) == (;x=2.0)
            @test extern(Composite{Tuple{Float64,}}(@thunk(0+2.0))) == (2.0,)
        end

        @testset "+ with other composites" begin
            @testset "Structs" begin
                @test (
                    Composite{Foo}(x=1.5) +
                    Composite{Foo}(x=2.5)
                ) == Composite{Foo}(x=4.0)
                @test (
                    Composite{Foo}(y=1.5) +
                    Composite{Foo}(x=2.5)
                ) == Composite{Foo}(y=1.5, x=2.5)
                @test (
                    Composite{Foo}(y=1.5, x=1.5) +
                    Composite{Foo}(x=2.5)
                    ) == Composite{Foo}(y=1.5, x=4.0)
            end

            @testset "Tuples" begin
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
        end

        @testset "+ with Primals" begin
            @testset "Structs" begin
                @test Foo(3.5, 1.5) + Composite{Foo}(x=2.5) == Foo(6.0, 1.5)
                @test Composite{Foo}(x=2.5) + Foo(3.5, 1.5) == Foo(6.0, 1.5)
            end

            @testset "Tuples" begin
                @test ((1.0, 2.0) + Composite{Tuple{Float64, Float64}}(1.0, 1.0)) == (2.0, 3.0)
                @test (Composite{Tuple{Float64, Float64}}(1.0, 1.0)) + (1.0, 2.0) == (2.0, 3.0)
            end

            @testset "NamedTuple" begin
                ntx = (; a=1.5)
                @test Composite{typeof(ntx)}(; ntx...) + ntx == (; a=3.0)

                nty = (; a=1.5, b=0.5)
                @test Composite{typeof(nty)}(; nty...) + nty == (; a=3.0, b=1.0)
            end

        end

        @testset "+ with Primals, with inner constructor" begin
            value = StructWithInvariant(10.0)
            diff = Composite{StructWithInvariant}(x=2.0, x2=6.0)
            @test_throws ChainRulesCore.PrimalAdditionFailedException (value + diff)
            @test_throws ChainRulesCore.PrimalAdditionFailedException (diff + value)

            # Now we define constuction for ChainRulesCore.jl's purposes:
            # It is going to determine the root quanity of the invarient
            function ChainRulesCore.construct(::Type{StructWithInvariant}, nt::NamedTuple)
                x = (nt.x + nt.x2/2)/2
                return StructWithInvariant(x)
            end
            @test value + diff == StructWithInvariant(12.5)
            @test diff + value == StructWithInvariant(12.5)
        end

        @testset "Scaling" begin
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
        end

        @testset "show" begin
            @test repr(Composite{Foo}(x=1,)) == "Composite{Foo}(x = 1,)"
            @test repr(Composite{Tuple{Int,Int}}(1, 2)) == "Composite{Tuple{Int64,Int64}}(1, 2)"
        end

        @testset "internals" begin
            @testset "Can't do backing on primative type" begin
                @test_throws Exception ChainRulesCore.backing(1.4)
            end

            @testset "Internals don't allocate a ton" begin
                bk = (; x=1.0, y=2.0)
                @test (@allocated(ChainRulesCore.construct(Foo, bk))) <= 32
                @test (@allocated ChainRulesCore.elementwise_add(bk, bk)) <= 48
            end
        end
    end

    @testset "Refine Differential" begin
        @test refine_differential(typeof(1.0 + 1im), Wirtinger(2,2)) == Wirtinger(2,2)
        @test refine_differential(typeof([1.0 + 1im]), Wirtinger(2,2)) == Wirtinger(2,2)

        @test refine_differential(typeof(1.2), Wirtinger(2,2)) == 4
        @test refine_differential(typeof([1.2]), Wirtinger(2,2)) == 4

        # For most differentials, in most domains, this does nothing
        for der in (DoesNotExist(), @thunk(23), @thunk(Wirtinger(2,2)), [1 2], One(), Zero(), 0.0)
            for 𝒟 in typeof.((1.0 + 1im, [1.0 + 1im], 1.2, [1.2]))
                @test refine_differential(𝒟, der) === der
            end
        end
    end
end
