@testset "Thunk" begin
    @test @thunk(3) isa Thunk

    @testset "==" begin
        @test @thunk(3.2) == InplaceableThunk(@thunk(3.2), x -> x + 3.2)
        @test @thunk(3.2) == 3.2
        @test 3.2 == InplaceableThunk(@thunk(3.2), x -> x + 3.2)
    end

    @testset "iterate" begin
        a = [1.0, 2.0, 3.0]
        t = @thunk(a)
        for (i, j) in zip(a, t)
            @test i == j
        end
    end

    @testset "show" begin
        rep = repr(Thunk(rand))
        @test occursin(r"Thunk\(.*rand.*\)", rep)
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
        x_complex = [(2.0 + im) 4.0; 8.0 (5.0 + 4im)]
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
                was_unthunked += 1;
                [1.0, 2.0, 3.0]
            end

            was_unthunked = 0
            @test array_thunk .+ fill(10, 3) .+  fill(10, 3) == [21.0, 22.0, 23.0]
            @test was_unthunked == 1

            was_unthunked = 0
            @test array_thunk .+ 10.0 .+ 10.0 == [21.0, 22.0, 23.0]
            @test was_unthunked == 1

        end

        @testset "Scalar" begin
            was_unthunked=0
            scalar_thunk = @thunk begin
                was_unthunked += 1;
                sqrt(4.0)
            end

            was_unthunked = 0
            @test scalar_thunk .+ fill(10, 3) .+  fill(10, 3) == [22.0, 22.0, 22.0]
            @test was_unthunked == 1

            was_unthunked = 0
            @test scalar_thunk .+ 10.0 .+ 10.0 == 22.0
            @test was_unthunked == 1
        end
    end

    @testset "basic math" begin
        @test 1 = - @thunk(-1)
        @test 1 = @thunk(2) - 1
        @test 1 = 2 - @thunk(1)
        @test 1.0 = @thunk(1) / 1.0
        @test 1.0 = 1.0 / @thunk(1)

        @test 1 = real(@thunk(1 + 1im))
        @test 1 = imag(@thunk(1 + 1im))
        @test 1 + 1im = Complex(@thunk(1 + 1im))
        @test 1 + 1im = Complex(@thunk(1), @thunk(1))
    end

    @testset "Base functions" begin
        @test Int64 === eltype(@thunk([1, 2]))
        @test 1.0 convert(Float64, @thunk(1))
        @test @thunk(1) == convert(Thunk, @thunk(1))

        @test 3 == mapreduce(_ -> 1, +, @thunk([1, 2, 3]))
        @test 3 == mapreduce((_, _) -> 1, +, [1, 2, 3], @thunk([1, 2, 3]))
        @test [4, 6] == sum!([1 1], @thunk([1 2; 3 4]))

        @test (2,) = size(@thunk([1, 2]))
        @test 2 = size(@thunk([1, 2]), 1)

        @test [1, 2] == vec(@thunk([1, 2])) 
        @test Base.OneTo(3) == axes(@thunk([1, 2, 3]), 1)
        @test [1; 2; 3] == reshape(@thunk([1, 2, 3]), 1, 3)
        @test 1.0 == getindex(@thunk([1.0, 2.0]), 1)
        @test [0.0, 2.0] == setindex!(@thunk([1.0, 2.0]), 0.0, 1)
        @test [4; 5; 6] == selectdim([1 2 3; 4 5 6], 1, 2)

    end
end
