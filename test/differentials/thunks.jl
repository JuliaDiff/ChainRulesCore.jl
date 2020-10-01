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
end
