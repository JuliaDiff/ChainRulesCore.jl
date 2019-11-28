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
