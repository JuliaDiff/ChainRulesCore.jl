@testset "NO_FIELDS" begin
    @test (@test_deprecated NO_FIELDS) isa NoTangent
end

@testset "extern" begin
    @test extern(@thunk(3)) == 3
    @test extern(@thunk(@thunk(3))) == 3

    @test extern(Tangent{Foo}(x=2.0)) == (;x=2.0)
    @test extern(Tangent{Tuple{Float64,}}(2.0)) == (2.0,)
    @test extern(Tangent{Dict}(Dict(4 => 3))) == Dict(4 => 3)

    # with differentials on the inside
    @test extern(Tangent{Foo}(x=@thunk(0+2.0))) == (;x=2.0)
    @test extern(Tangent{Tuple{Float64,}}(@thunk(0+2.0))) == (2.0,)
    @test extern(Tangent{Dict}(Dict(4 => @thunk(3)))) == Dict(4 => 3)

    z = ZeroTangent()
    @test extern(z) === false
    dne = NoTangent()
    @test_throws Exception extern(dne)
    E = ChainRulesCore.NotImplementedException
    @test_throws E extern(ni)
end


@testset "Deprecated: calling thunks should call inner function" begin
    @test_deprecated (@thunk(3))() == 3
    @test_deprecated (@thunk(@thunk(3)))() isa Thunk
end
