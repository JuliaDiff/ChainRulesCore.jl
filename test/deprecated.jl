@testset "NO_FIELDS" begin
    # Following doesn't work because of some deprecate_binding weirdness with not printing
    # @test (@test_deprecated NO_FIELDS) isa NoTangent
    # So just test it gives the old behavour
    @test NO_FIELDS isa NoTangent
end

@testset "extern" begin
    @test (@test_deprecated extern(@thunk(3))) == 3
    @test (@test_deprecated extern(@thunk(@thunk(3)))) == 3

    @test (@test_deprecated extern(Tangent{Foo}(x=2.0))) == (;x=2.0)
    @test (@test_deprecated extern(Tangent{Tuple{Float64,}}(2.0))) == (2.0,)
    @test (@test_deprecated extern(Tangent{Dict}(Dict(4 => 3)))) == Dict(4 => 3)

    # with differentials on the inside
    @test (@test_deprecated extern(Tangent{Foo}(x=@thunk(0+2.0)))) == (;x=2.0)
    @test (@test_deprecated extern(Tangent{Tuple{Float64,}}(@thunk(0+2.0)))) == (2.0,)
    @test (@test_deprecated extern(Tangent{Dict}(Dict(4 => @thunk(3))))) == Dict(4 => 3)

    z = ZeroTangent()
    @test (@test_deprecated extern(z)) === false
    
    # @test_throws doesn't play nice with `@test_deprecated` so have to be loud
    dne = NoTangent()
    @test_throws Exception extern(dne)
    ni = @not_implemented("no")
    @test_throws ChainRulesCore.NotImplementedException extern(ni)
end


@testset "Deprecated: calling thunks should call inner function" begin
    @test (@test_deprecated (@thunk(3))()) == 3
    @test (@test_deprecated (@thunk(@thunk(3)))()) isa Thunk
end

@testset "Deprecated: Inplacable Thunk argument order" begin
    @test (@test_deprecated InplaceableThunk(@thunk([1]), x->x.+=1)) isa InplaceableThunk
end

@testset "Deprecated: convert from Tangent" begin
    @test convert(NamedTuple, Tangent{Foo}(x=2.5)) == (; x=2.5)
    @test convert(Tuple, Tangent{Tuple{Float64,}}(2.0)) == (2.0,)
    @test convert(Dict, Tangent{Dict}(Dict(4 => 3))) == Dict(4 => 3)
end
