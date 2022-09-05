@testset "accumulation.jl" begin
    @testset "is_inplaceable_destination" begin
        is_inplaceable_destination = ChainRulesCore.is_inplaceable_destination

        @test is_inplaceable_destination([1.0, 2.0, 3.0])
        @test !is_inplaceable_destination([1, 2, 3, 4])  # gradients cannot reliably be written into integer arrays
        @test !is_inplaceable_destination(1:4.0)

        @test is_inplaceable_destination(Diagonal([1.0, 2.0, 3.0]))
        @test !is_inplaceable_destination(Diagonal(1:4.0))

        @test is_inplaceable_destination(view([1.0, 2.0, 3.0], :, :))
        @test is_inplaceable_destination(view([1.0 2.0; 3.0 4.0], :, 2))
        @test !is_inplaceable_destination(view(1:4.0, :, :))
        mat = view([1.0, 2.0, 3.0], :, fill(1, 10))
        @test !is_inplaceable_destination(mat)  # The concern is that `mat .+= x` is unsafe on GPU / parallel.

        @test !is_inplaceable_destination(falses(4))  # gradients can never be written into boolean
        @test is_inplaceable_destination(spzeros(4))
        @test is_inplaceable_destination(spzeros(2, 2))

        @test !is_inplaceable_destination(1:3.0)
        @test !is_inplaceable_destination(@SVector [1.0, 2.0, 3.0])
        @test !is_inplaceable_destination(Hermitian([1.0 2.0; 2.0 4.0]))
    end

    @testset "add!!" begin
        @testset "scalar" begin
            @test 16 == add!!(12, 4)
        end

        @testset "misc AbstractTangent subtypes" begin
            @test 16 == add!!(12, @thunk(2 * 2))
            @test 16 == add!!(16, ZeroTangent())

            @test 16 == add!!(16, NoTangent())  # Should this be an error?
        end

        @testset "add!!(::AbstractArray, ::AbstractArray)" begin
            @testset "LHS Array (inplace)" begin
                @testset "RHS Array" begin
                    A = [1.0 2.0; 3.0 4.0]
                    accumuland = -1.0 * ones(2, 2)
                    ret = add!!(accumuland, A)
                    @test ret === accumuland  # must be same object
                    @test accumuland == [0.0 1.0; 2.0 3.0]
                end

                @testset "RHS StaticArray" begin
                    A = @SMatrix [1.0 2.0; 3.0 4.0]
                    accumuland = -1.0 * ones(2, 2)
                    ret = add!!(accumuland, A)
                    @test ret === accumuland  # must be same object
                    @test accumuland == [0.0 1.0; 2.0 3.0]
                end

                @testset "RHS Diagonal" begin
                    A = Diagonal([1.0, 2.0])
                    accumuland = -1.0 * ones(2, 2)
                    ret = add!!(accumuland, A)
                    @test ret === accumuland  # must be same object
                    @test accumuland == [0.0 -1.0; -1.0 1.0]
                end
            end

            @testset "add!!(::StaticArray, ::Array) (out of place)" begin
                A = [1.0 2.0; 3.0 4.0]
                accumuland = @SMatrix [-1.0 -1.0; -1.0 -1.0]
                ret = add!!(accumuland, A)
                @test ret == [0.0 1.0; 2.0 3.0]  # must return right answer
                @test ret !== accumuland  # must not be same object
                @test accumuland == [-1.0 -1.0; -1.0 -1.0]  # must not have changed
            end

            @testset "add!!(::Diagonal{<:Vector}, ::Diagonal{<:Vector}) (inplace)" begin
                A = Diagonal([1.0, 2.0])
                accumuland = Diagonal([-2.0, -2.0])
                ret = add!!(accumuland, A)
                @test ret === accumuland  # must be same object
                @test accumuland == Diagonal([-1.0, 0.0])
            end

            @testset "Unhappy Path" begin
                # wrong length
                @test_throws DimensionMismatch add!!(ones(4, 4), ones(2, 2))
                # wrong shape
                @test_throws DimensionMismatch add!!(ones(4, 4), ones(16))
                # wrong type (adding scalar to array)
                @test_throws MethodError add!!(ones(4), 21.0)
            end
        end

        @testset "AbstractThunk $(typeof(thunk))" for thunk in (
            @thunk(-1.0 * ones(2, 2)),
            InplaceableThunk(x -> x .-= ones(2, 2), @thunk(-1.0 * ones(2, 2))),
        )
            @testset "in place" begin
                accumuland = [1.0 2.0; 3.0 4.0]
                ret = add!!(accumuland, thunk)
                @test ret == [0.0 1.0; 2.0 3.0]  # must return right answer
                @test ret === accumuland  # must be same object
            end

            @testset "out of place" begin
                accumuland = @SMatrix [1.0 2.0; 3.0 4.0]

                ret = add!!(accumuland, thunk)
                @test ret == [0.0 1.0; 2.0 3.0]  # must return right answer
                @test ret !== accumuland  # must not be same object
                @test accumuland == [1.0 2.0; 3.0 4.0]  # must not have mutated
            end
        end

        @testset "not actually inplace but said it was" begin
            # thunk should never be used in this test
            ithunk = InplaceableThunk(@thunk(@assert false)) do x
                77 * ones(2, 2)  # not actually inplace (also wrong)
            end
            accumuland = ones(2, 2)
            @assert ChainRulesCore.debug_mode() == false
            # without debug being enabled should return the result, not error
            @test 77 * ones(2, 2) == add!!(accumuland, ithunk)

            ChainRulesCore.debug_mode() = true  # enable debug mode
            # with debug being enabled should error
            @test_throws ChainRulesCore.BadInplaceException add!!(accumuland, ithunk)
            ChainRulesCore.debug_mode() = false  # disable it again
        end
    end

    @testset "showerror BadInplaceException" begin
        BadInplaceException = ChainRulesCore.BadInplaceException
        ithunk = InplaceableThunk(x̄ -> nothing, @thunk(@assert false))
        msg = sprint(showerror, BadInplaceException(ithunk, [22], [23]))
        @test occursin("22", msg)

        msg_equal = sprint(showerror, BadInplaceException(ithunk, [22], [22]))
        @test occursin("equal", msg_equal)
    end
end
