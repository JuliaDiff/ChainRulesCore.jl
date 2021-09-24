@testset "ignore_gradients.jl" begin
    f() = return 4.0

    y, ẏ = frule((1.0, ), ignore_gradients, f)
    @test y == f()
    @test ẏ == NoTangent()

    y, pb = rrule(ignore_gradients, f)
    @test y == f()
    @test pb(1.0) == (NoTangent(), NoTangent())
end
