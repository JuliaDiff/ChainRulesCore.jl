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
        @test transpose(w) == w
    end

    @testset "Casted" begin
        f, x, y = +, 2.1, [1, 2, 3]
        c = cast(f, x, y)
        ext = f.(x, y)
        @test extern(c) == ext
        @test extern(c + 2) == ext .+ 2
        @test extern(c + Zero()) == ext
        @test extern(c + One()) == ext .+ true
        @test extern(c + c) == ext .+ ext
        @test extern(3 * c) == 3 .* ext
        @test extern(Zero() * c) == false .* c
        @test extern(One() * c) == ext
        @test extern(c * c) == ext .* ext
        for (i,c_i) in enumerate(c)
            @test c_i == ext[i]
        end
        @test broadcastable(c) == Broadcast.broadcasted(f, x, y)
        @test extern(conj(c)) == ext
        @test extern(conj(im * c)) == -im .* ext
        @test extern(transpose(c)) == transpose(ext)
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
        @test transpose(z) == z
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
        @test transpose(o) == o
    end

    @testset "Thunk" begin
        f, x = transpose, [1, 2, 3]
        t = @thunk(f(x))
        ext = f(x)
        @test extern(t) == ext
        @test extern(t + Zero()) == ext
        @test extern(t + t) == ext + ext
        @test extern(3 * t) == 3 * ext
        @test Zero() * t == Zero()
        @test extern(One() * t) == ext
        for (i,t_i) in enumerate(t)
            @test t_i == ext[i]
        end
        @test broadcastable(t) == broadcastable(ext)
        @test extern(conj(t)) == ext
        @test extern(conj(im * t)) == -im .* ext
        @test extern(transpose(t)) == x
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
end
