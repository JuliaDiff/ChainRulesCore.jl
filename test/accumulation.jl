@testset "accumulation.jl" begin
    @testset "scalar" begin
        @test 16 == accumulate!!(12, 4)
    end

    @testset "Differentials" begin
        @test 16 == accumulate!!(12, @thunk(2*2))
        @test 16 == accumulate!!(16, Zero())

        @test 16 == accumulate!!(16, DoesNotExist())  # Should this be an error?
    end

    @testset "Array" begin
        @testset "Happy Path" begin
            @testset "RHS Array" begin
                A = [1.0 2.0; 3.0 4.0]
                result = -1.0*ones(2,2)
                ret = accumulate!!(result, A)
                @test ret === result  # must be same object
                @test result == [0.0 1.0; 2.0 3.0]
            end

            @testset "RHS StaticArray" begin
                A = @SMatrix [1.0 2.0; 3.0 4.0]
                result = -1.0*ones(2,2)
                ret = accumulate!!(result, A)
                @test ret === result  # must be same object
                @test result == [0.0 1.0; 2.0 3.0]
            end

            @testset "RHS Diagonal" begin
                A=Diagonal([1.0, 2.0])
                result = -1.0*ones(2,2)
                ret = accumulate!!(result, A)
                @test ret === result  # must be same object
                @test result == [0.0 -1.0; -1.0 1.0]
            end
        end

        @testset "Unhappy Path" begin
            # wrong length
            @test_throws DimensionMismatch accumulate!!(ones(4,4), ones(2,2))
            # wrong shape
            @test_throws DimensionMismatch accumulate!!(ones(4,4), ones(16))
            # wrong type (adding scalar to array)
            @test_throws MethodError accumulate!!(ones(4), 21.0)
        end
    end

    @testset "InplaceableThunk" begin
        A=[1.0 2.0; 3.0 4.0]
        ithunk = InplaceableThunk(
            @thunk(A*B),
            x -> x.+=A
        )

        result = -1.0*ones(2,2)
        ret = accumulate!!(result, ithunk)
        @test ret === result  # must be same object
        @test result == [0.0 1.0; 2.0 3.0]
    end
end
