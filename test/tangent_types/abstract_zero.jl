@testset "AbstractZero" begin
    @testset "iszero" begin
        @test iszero(ZeroTangent())
        @test iszero(NoTangent())
    end

    @testset "Linear operators" begin
        @test view(ZeroTangent(), 1) == ZeroTangent()
        @test view(NoTangent(), 1, 2) == NoTangent()

        @test sum(ZeroTangent()) == ZeroTangent()
        @test sum(NoTangent(); dims=2) == NoTangent()

        @test reshape(NoTangent(), (1, :)) === NoTangent()
        @test reshape(ZeroTangent(), 2, 3, 4, 5) === ZeroTangent()

        @test reverse(NoTangent()) === NoTangent()
        @test reverse(ZeroTangent()) === ZeroTangent()
        @test reverse(ZeroTangent(); dims=1) === ZeroTangent()
        @test reverse(ZeroTangent(), 2, 5) === ZeroTangent()
    end

    @testset "ZeroTangent" begin
        z = ZeroTangent()
        @test z + z === z
        @test z + 1 === 1
        @test 1 + z === 1
        @test z - z === z
        @test z - 1 === -1
        @test 1 - z === 1
        @test -z === z
        @test z * z === z
        @test z * 11.1 === ZeroTangent()
        @test 12.3 * z === ZeroTangent()
        @test dot(z, z) === z
        @test dot(z, 1.8) === z
        @test dot(2.1, z) === z
        @test dot([1, 2], z) === z
        @test dot(z, [1, 2]) === z
        for x in z
            @test x === z
        end
        @test broadcastable(z) isa Ref{ZeroTangent}
        @test zero(@thunk(3)) === z
        @test zero(NoTangent()) === z
        @test zero(ZeroTangent) === z
        @test zero(NoTangent) === z
        @test zero(Tangent{Tuple{Int,Int}}((1, 2))) === z
        for f in (transpose, adjoint, conj)
            @test f(z) === z
        end
        @test z / 2 === z / [1, 3] === z

        @test eltype(z) === ZeroTangent
        @test eltype(ZeroTangent) === ZeroTangent

        # use mutable objects to test the strong `===` condition
        x = ones(2)
        @test muladd(ZeroTangent(), 2, x) === x
        @test muladd(2, ZeroTangent(), x) === x
        @test muladd(ZeroTangent(), ZeroTangent(), x) === x
        @test muladd(2, 2, ZeroTangent()) === 4
        @test muladd(x, ZeroTangent(), ZeroTangent()) === ZeroTangent()
        @test muladd(ZeroTangent(), x, ZeroTangent()) === ZeroTangent()
        @test muladd(ZeroTangent(), ZeroTangent(), ZeroTangent()) === ZeroTangent()

        @test reim(z) === (ZeroTangent(), ZeroTangent())
        @test real(z) === ZeroTangent()
        @test imag(z) === ZeroTangent()

        @test complex(z) === z
        @test complex(z, z) === z
        @test complex(z, 2.0) === Complex{Float64}(0.0, 2.0)
        @test complex(1.5, z) === Complex{Float64}(1.5, 0.0)
        @test Complex(z, 2.0) === Complex{Float64}(0.0, 2.0)
        @test Complex(1.5, z) === Complex{Float64}(1.5, 0.0)
        @test ComplexF64(z, 2.0) === Complex{Float64}(0.0, 2.0)
        @test ComplexF64(1.5, z) === Complex{Float64}(1.5, 0.0)

        @test convert(Bool, ZeroTangent()) === false
        @test convert(Int64, ZeroTangent()) === Int64(0)
        @test convert(Float32, ZeroTangent()) === 0.0f0
        @test convert(ComplexF64, ZeroTangent()) === 0.0 + 0.0im

        @test z[1] === z
        @test z[1:3] === z
        @test z[1, 2] === z
        @test getindex(z) === z

        @test z.foo === z
        
        @test first(z) === z
        @test last(z) === z
        @test Base.tail(z) === z
    end

    @testset "NoTangent" begin
        dne = NoTangent()
        @test dne + dne == dne
        @test dne + 1 == 1
        @test 1 + dne == 1
        @test dne - dne == dne
        @test dne - 1 == -1
        @test 1 - dne == 1
        @test -dne == dne
        @test dne * dne == dne
        @test dne * 11.1 == dne
        @test 12.1 * dne == dne
        @test dot(dne, dne) == dne
        @test dot(dne, 17.2) == dne
        @test dot(11.9, dne) == dne

        @test ZeroTangent() + dne == dne
        @test dne + ZeroTangent() == dne
        @test ZeroTangent() - dne == dne
        @test dne - ZeroTangent() == dne

        @test ZeroTangent() * dne == ZeroTangent()
        @test dne * ZeroTangent() == ZeroTangent()
        @test dot(ZeroTangent(), dne) == ZeroTangent()
        @test dot(dne, ZeroTangent()) == ZeroTangent()

        @test norm(ZeroTangent()) == 0
        @test norm(ZeroTangent(), 0.4) == 0

        for x in dne
            @test x === dne
        end
        @test broadcastable(dne) isa Ref{NoTangent}
        for f in (transpose, adjoint, conj)
            @test f(dne) === dne
        end
        @test dne / 2 === dne / [1, 3] === dne

        @test convert(Int64, NoTangent()) == 0
        @test convert(Float64, NoTangent()) == 0.0

        @test dne[1] === dne
        @test dne[1:3] === dne
        @test dne[1, 2] === dne
        @test getindex(dne) === dne
        @test dne.foo === dne
    end

    @testset "ambiguities" begin
        M = @eval module M
            using ChainRulesCore

            struct X{R,S} <: Number
                a::R
                b::S
                hasvalue::Bool

                function X{R,S}(a, b, hv=true) where {R,S}
                    isa(hv, Bool) || error("must be bool")
                    return new{R,S}(a, b, hv)
                end
            end
        end
        @test isempty(detect_ambiguities(M))
    end
end

@testset "zero_tangent" begin
    @testset "basics" begin
        @test zero_tangent(1) === 0
        @test zero_tangent(1.0) === 0.0
        @test zero_tangent(true) === NoTangent()

        mutable struct MutDemo
            x::Float64
        end
        struct Demo
            x::Float64
        end
        @test zero_tangent(MutDemo(1.5)) isa MutableTangent{MutDemo}
        @test iszero(zero_tangent(MutDemo(1.5)))

        @test zero_tangent((; a=1)) isa Tangent{typeof((; a = 1))}
        @test zero_tangent(Demo(1.2)) isa Tangent{Demo}
        @test zero_tangent(Demo(1.2)).x === 0.0

        @test zero_tangent([1.0, 2.0]) == [0.0, 0.0]
        @test zero_tangent([[1.0, 2.0], [3.0]]) == [[0.0, 0.0], [0.0]]

        @test zero_tangent((1.0, 2.0)) == Tangent{Tuple{Float64,Float64}}(0.0, 0.0)

        @test ==(
            zero_tangent(Dict{Int, Float64}(1 => 2.4)),
             Tangent{Dict{Int,Float64}}(Dict{Int, Float64}())
        )
        if isdefined(Base, :PersistentDict)
            @test ==(
                zero_tangent(Base.PersistentDict(1 => 2.4)),
                Tangent{Base.PersistentDict{Int,Float64}}(Dict{Int, Float64}())
            )
        end


        # Higher order
        # StructuralTangents are valid tangents for themselves (just like Numbers)
        # and indeed we prefer that, otherwise higher order structural tangents are kinda
        # nightmarishly complex types.
        @test zero_tangent(zero_tangent(Demo(1.5))) == zero_tangent(Demo(1.5))
        @test zero_tangent(zero_tangent((1.5, 2.5))) == Tangent{Tuple{Float64, Float64}}(0.0, 0.0)
        @test zero_tangent(zero_tangent(MutDemo(1.5))) == zero_tangent(MutDemo(1.5))
    end

    @testset "Weird types" begin
        @test iszero(zero_tangent(typeof(Int)))  # primative type
        @test iszero(zero_tangent(typeof(Base.RefValue)))  # struct
        @test iszero(zero_tangent(Vector))  # UnionAll
        @test iszero(zero_tangent(Union{Int, Float64}))  # Union
        @test iszero(zero_tangent(:abc))
        @test iszero(zero_tangent("abc"))
        @test iszero(zero_tangent(sin))

        @test iszero(zero_tangent(:(1 + 1)))
    end

    @testset "undef elements Vector" begin
        x = Vector{Vector{Float64}}(undef, 3)
        x[2] = [1.0, 2.0]
        dx = zero_tangent(x)
        @test dx isa Vector{Vector{Float64}}
        @test length(dx) == 3
        @test !isassigned(dx, 1)  # We may reconsider this later
        @test dx[2] == [0.0, 0.0]
        @test !isassigned(dx, 3)  # We may reconsider this later

        a = Vector{MutDemo}(undef, 3)
        a[2] = MutDemo(1.5)
        da = zero_tangent(a)
        @test !isassigned(da, 1)  # We may reconsider this later
        @test iszero(da[2])
        @test !isassigned(da, 3)  # We may reconsider this later

        db = zero_tangent(Vector{MutDemo}(undef, 3))
        @test all(ii -> !isassigned(db, ii), eachindex(db))  # We may reconsider this later
        @test length(db) == 3
        @test db isa Vector
    end

    @testset "undef fields struct" begin
        dx = zero_tangent(Core.Box())
        @test dx.contents isa ZeroTangent
        @test (dx.contents = 2.0) == 2.0  # should be assignable

        mutable struct MyPartiallyDefinedStruct
            intro::Float64
            contents::Number
            MyPartiallyDefinedStruct(x) = new(x)
        end
        dy = zero_tangent(MyPartiallyDefinedStruct(1.5))
        @test iszero(dy.intro)
        @test iszero(dy.contents)
        @test (dy.contents = 2.0) == 2.0  # should be assignable

        mutable struct MyPartiallyDefinedStructWithAnys
            intro::Float64
            contents::Any
            MyPartiallyDefinedStructWithAnys(x) = new(x)
        end
        dy = zero_tangent(MyPartiallyDefinedStructWithAnys(1.5))
        @test iszero(dy.intro)
        @test iszero(dy.contents)
        @test dy.contents === ZeroTangent()  # we just don't know anything about this data
        @test (dy.contents = 2.0) == 2.0  # should be assignable
        @test (dy.contents = [2.0, 4.0]) == [2.0, 4.0]  # should be assignable to different values

        mutable struct MyStructWithNonConcreteFields
            x::Any
            y::Union{Float64,Vector{Float64}}
            z::AbstractVector
        end
        d = zero_tangent(MyStructWithNonConcreteFields(1.0, 2.0, [3.0]))
        @test iszero(d.x)
        d.x = Tangent{Base.RefValue{Float64}}(; x=1.5)
        @test d.x == Tangent{Base.RefValue{Float64}}(; x=1.5)  #should be assignable
        d.x = 2.4
        @test d.x == 2.4  #should be assignable
        @test iszero(d.y)
        d.y = 2.4
        @test d.y == 2.4  #should be assignable
        d.y = [2.4]
        @test d.y == [2.4]  #should be assignable
        @test iszero(d.z)
        d.z = [1.0, 2.0]
        @test d.z == [1.0, 2.0]
        d.z = @view [2.0, 3.0, 4.0][1:2]
        @test d.z == [2.0, 3.0]
        @test d.z isa SubArray
    end


    @testset "aliasing" begin
        a = Base.RefValue(1.5)
        b = (a, 1.0, a)
        db = zero_tangent(b)
        @test iszero(db)
        @test_broken db[1] === db[3]
        @test db[2] == 0.0

        x = [1.5]
        y = [x, [1.0], x]
        dy = zero_tangent(y)
        @test iszero(dy)
        @test_broken dy[1] === dy[3]
        @test dy[2] == [0.0]
    end

    @testset "cyclic references" begin
        mutable struct Link
            data::Float64
            next::Link
            Link(data) = new(data)
        end

        lk = Link(1.5)
        lk.next = lk

        @test_broken d = zero_tangent(lk)
        @test_broken d.data == 0.0
        @test_broken d.next === d

        struct CarryingArray
            x::Vector
        end
        ca = CarryingArray(Any[1.5])
        push!(ca.x, ca)
        @test_broken d_ca = zero_tangent(ca)
        @test_broken d_ca[1] == 0.0
        @test_broken d_ca[2] === _ca

        # Idea: check if typeof(xs) <: eltype(xs), if so need to cache it before computing
        xs = Any[1.5]
        push!(xs, xs)
        @test_broken d_xs = zero_tangent(xs)
        @test_broken d_xs[1] == 0.0
        @test_broken d_xs[2] == d_xs
    end
end
