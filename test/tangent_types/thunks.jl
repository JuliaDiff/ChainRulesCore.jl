@testset "Thunk" begin
    MutateThunkException = ChainRulesCore.MutateThunkException

    @test @thunk(3) isa Thunk

    @testset "==" begin
        @test @thunk(3.2) == InplaceableThunk(x -> x + 3.2, @thunk(3.2))
    end

    @testset "iterate" begin
        a = [1.0, 2.0, 3.0]
        t = @thunk(a)
        for (i, j) in zip(a, t)
            @test i == j
        end

        @test nothing === iterate(@thunk ()) == iterate(())
    end
    
    @testset "first, last, tail" begin
        @test first(@thunk (1,2,3) .+ 4) === 5
        @test last(@thunk (1,2,3) .+ 4) === 7
        @test Base.tail(@thunk (1,2,3) .+ 4) === (6, 7)
        @test Base.tail(@thunk NoTangent() * 5) === NoTangent()
    end

    @testset "show" begin
        rep = repr(Thunk(rand))
        @test occursin(r"Thunk\(.*rand.*\)", rep)
    end

    @testset "convert" begin
        @test convert(Thunk, ZeroTangent()) isa Thunk
    end

    @testset "unthunk" begin
        @test unthunk(@thunk(3)) == 3
        @test unthunk(@thunk(@thunk(3))) isa Thunk
    end

    @testset "erroring thunks should include the source in the backtrack" begin
        expected_line = (@__LINE__) + 2  # for testing it is at right palce
        try
            x = @thunk(error())
            unthunk(x)
        catch err
            err isa ErrorException || rethrow()
            st = stacktrace(catch_backtrace())
            # Should be 2nd last line, as last line will be the `error` function
            stackframe = st[2]
            @test stackframe.line == expected_line
            @test stackframe.file == Symbol(@__FILE__)
        end
    end

    @testset "Linear operators" begin
        x_real = [2.0 4.0; 8.0 5.0]
        x_complex = [(2.0+im) 4.0; 8.0 (5.0+4im)]
        @testset "$(typeof(x))" for x in (x_real, x_complex)
            x_thunked = @thunk(1.0 * x)
            @test unthunk(x_thunked') == x'
            @test unthunk(transpose(x_thunked)) == transpose(x)
        end
    end

    @testset "Broadcast" begin
        @testset "Array" begin
            was_unthunked = 0
            array_thunk = @thunk begin
                was_unthunked += 1
                [1.0, 2.0, 3.0]
            end

            was_unthunked = 0
            @test array_thunk .+ fill(10, 3) .+ fill(10, 3) == [21.0, 22.0, 23.0]
            @test was_unthunked == 1

            was_unthunked = 0
            @test array_thunk .+ 10.0 .+ 10.0 == [21.0, 22.0, 23.0]
            @test was_unthunked == 1
        end

        @testset "Scalar" begin
            was_unthunked = 0
            scalar_thunk = @thunk begin
                was_unthunked += 1
                sqrt(4.0)
            end

            was_unthunked = 0
            @test scalar_thunk .+ fill(10, 3) .+ fill(10, 3) == [22.0, 22.0, 22.0]
            @test was_unthunked == 1

            was_unthunked = 0
            @test scalar_thunk .+ 10.0 .+ 10.0 == 22.0
            @test was_unthunked == 1
        end
    end

    @testset "basic math" begin
        @test 1 == -@thunk(-1)
        @test 1 == @thunk(2) - 1
        @test 1 == 2 - @thunk(1)
        @test 1 == @thunk(2) - @thunk(1)
        @test 1.0 == @thunk(1) / 1.0
        @test 1.0 == 1.0 / @thunk(1)
        @test 1 == @thunk(1) / @thunk(1)

        # check method ambiguities (#589)
        for a in (ZeroTangent(), NoTangent())
            @test a / @thunk(2) === a
        end

        @test 1 == real(@thunk(1 + 1im))
        @test 1 == imag(@thunk(1 + 1im))
        @test 1 + 1im == Complex(@thunk(1 + 1im))
        @test 1 + 1im == Complex(@thunk(1), @thunk(1))
    end

    @testset "Base functions" begin
        v = [1, 2, 3]
        t = @thunk(v)

        m = rand(3, 3)
        tm = @thunk(m)

        if VERSION >= v"1.2"
            @test 3 == mapreduce(_ -> 1, +, t)
            @test 3 == mapreduce((_, _) -> 1, +, v, t)
        end
        @test 10 == sum(@thunk([1 2; 3 4]))
        @test [4 6] == sum!([1 1], @thunk([1 2; 3 4]))

        @test fill(3.2, 3) == fill(@thunk(3.2), 3)
        @test v == vec(t)
        @test [1 2 3] == reshape(t, 1, 3)
        @test 1 == getindex(t, 1)
        @test_throws MutateThunkException setindex!(t, 0.0, 1)
        @test [4; 5; 6] == selectdim([1 2 3; 4 5 6], 1, 2)

        @test reverse(t) == reverse(v)
        @test reverse(t, 2) == reverse(v, 2)
        @test reverse(tm; dims=2) == reverse(m; dims=2)
    end

    @testset "LinearAlgebra" begin
        v = [1.0, 2.0, 3.0]
        tv = @thunk(v)
        a = [1.0 2.0; 3.0 4.0]
        t = @thunk(a)
        @test Array(a) == Array(t)
        @test Matrix(a) == Matrix(t)
        @test Diagonal(a) == Diagonal(t)
        @test LowerTriangular(a) == LowerTriangular(t)
        @test UpperTriangular(a) == UpperTriangular(t)
        @test Symmetric(a) == Symmetric(t)
        @test Hermitian(a) == Hermitian(t)

        if VERSION >= v"1.2"
            @test diagm(0 => v) == diagm(0 => tv)
            @test diagm(3, 4, 0 => v) == diagm(3, 4, 0 => tv)
            # Check against accidential type piracy
            # https://github.com/JuliaDiff/ChainRulesCore.jl/issues/472
            @test Base.which(diagm, Tuple{}()).module != ChainRulesCore
            @test Base.which(diagm, Tuple{Int,Int}).module != ChainRulesCore
        end
        @test tril(a) == tril(t)
        @test tril(a, 1) == tril(t, 1)
        @test triu(a) == triu(t)
        @test triu(a, 1) == triu(t, 1)
        @test tr(a) == tr(t)
        @test cross(v, v) == cross(v, tv)
        @test cross(v, v) == cross(tv, v)
        @test cross(v, v) == cross(tv, tv)
        @test dot(v, v) == dot(v, tv)
        @test dot(v, v) == dot(tv, v)
        @test dot(v, v) == dot(tv, tv)

        if VERSION >= v"1.2"
            @test_throws MutateThunkException ldiv!(2.0, deepcopy(t)) ==
                                              ldiv!(2.0, deepcopy(a))
            @test_throws MutateThunkException rdiv!(deepcopy(t), 2.0) ==
                                              rdiv!(deepcopy(a), 2.0)
        end

        @test mul!(deepcopy(a), a, a) == mul!(deepcopy(a), t, a)

        res = mul!(deepcopy(a), a, a, true, true)
        @test_throws MutateThunkException mul!(deepcopy(t), a, a, true, true)
        @test_throws MutateThunkException mul!(deepcopy(t), t, a, true, true)
        @test_throws MutateThunkException mul!(deepcopy(t), a, t, true, true)
        @test_throws MutateThunkException mul!(deepcopy(t), t, t, true, true)
        @test res == mul!(deepcopy(a), t, a, true, true)
        @test res == mul!(deepcopy(a), a, t, true, true)
        @test res == mul!(deepcopy(a), t, t, true, true)

        m = rand(3, 3)
        @test ger!(1.0, v, v, deepcopy(m)) == ger!(1.0, tv, v, deepcopy(m))
        @test ger!(1.0, v, v, deepcopy(m)) == ger!(1.0, v, tv, deepcopy(m))
        @test gemv!('C', 1.0, m, v, 1.0, deepcopy(v)) ==
              gemv!('C', 1.0, m, tv, 1.0, deepcopy(v))
        @test gemv('N', 1.0, m, v) == gemv('N', 1.0, m, tv)

        @test scal!(2, 2.0, v, 1) == scal!(2, @thunk(2.0), v, 1)
        @test_throws MutateThunkException LAPACK.trsyl!('C', 'C', m, m, @thunk(m))
    end
    
    @testset "printing" begin
        @test !contains(sprint(show, @thunk 1+1), "...")  # short thunks not abbreviated
        th = let x = rand(100)
            @thunk x .+ x'
        end
        @test contains(sprint(show, th), "...")  # but long ones are
        
        @test contains(sprint(show, InplaceableThunk(mul!, th)), "mul!")  # named functions left in InplaceableThunk
        str = sprint(show, InplaceableThunk(z -> z .+ ones(100), th))
        @test length(findall("...", str)) == 2  # now both halves shortened
    end
end
