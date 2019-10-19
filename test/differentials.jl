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

        @testset "calling thunks should call inner function" begin
            @test (@thunk(3))() == 3
            @test (@thunk(@thunk(3)))() isa Thunk
        end
    end

    @testset "No ambiguities in $f" for f in (+, *)
        # We don't use `Test.detect_ambiguities` as we are only interested in
        # the +, and * operations. We also would catch any that are unrelated
        # to this package. but that is not a problem. Since no such failings
        # occur in our dependencies.

        ambig_methods = [
            (m1, m2) for m1 in methods(f), m2 in methods(f) if Base.isambiguous(m1, m2)
        ]
        @test isempty(ambig_methods)
    end


    @testset "Refine Differential" begin
        for (p, c) in (
                       (2, -3),
                       (2.0 + im, 5.0 - 3.0im),
                       ([1+im, 2-im], [-3+im, 4+im]),
                       (@thunk(1+2), @thunk(4-3)),
                      )
            w = Wirtinger(p, c)
            @testset "$w" begin
                @test refine_differential(typeof(1.0 + 1im), w) === w
                @test refine_differential(typeof([1.0 + 1im]), w) === w

                @test refine_differential(typeof(1.2), w) == p + c
                @test refine_differential(typeof([1.2]), w) == p + c
            end

            g = ComplexGradient(c)
            @testset "$g" begin
                @test refine_differential(typeof(1.0 + 1im), g) === g
                @test refine_differential(typeof([1.0 + 1im]), g) === g

                @test refine_differential(typeof(1.2), g) == real(c)
                @test refine_differential(typeof([1.2]), g) == real(c)
            end
        end

        # For most differentials, in most domains, this does nothing
        for der in (DNE(), @thunk(23), [1 2], One(), Zero(), 0.0)
            for 𝒟 in typeof.((1.0 + 1im, [1.0 + 1im], 1.2, [1.2]))
                @test refine_differential(𝒟, der) === der
            end
        end
    end
end
