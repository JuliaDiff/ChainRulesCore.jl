using ChainRulesCore, Test
using LinearAlgebra, SparseArrays
using OffsetArrays, BenchmarkTools

# Like ForwardDiff.jl's Dual
struct Dual{T<:Real} <: Real
    value::T
    partial::T
end
Base.real(x::Dual) = x
Base.float(x::Dual) = Dual(float(x.value), float(x.partial))
Base.zero(x::Dual) = Dual(zero(x.value), zero(x.partial))

@testset "projection" begin

    #####
    ##### `Base`
    #####

    @testset "Base: numbers" begin
        # real / complex
        @test ProjectTo(1.0)(2.0 + 3im) === 2.0
        @test ProjectTo(1.0 + 2.0im)(3.0) === 3.0 + 0.0im
        @test ProjectTo(2.0+3.0im)(1+1im) === 1.0+1.0im
        @test ProjectTo(2.0)(1+1im) === 1.0
        

        # storage
        @test ProjectTo(1)(pi) === pi
        @test ProjectTo(1 + im)(pi) === ComplexF64(pi)
        @test ProjectTo(1//2)(3//4) === 3//4
        @test ProjectTo(1.0f0)(1 / 2) === 0.5f0
        @test ProjectTo(1.0f0 + 2im)(3) === 3.0f0 + 0im
        @test ProjectTo(big(1.0))(2) === 2
        @test ProjectTo(1.0)(2) === 2.0
    end

    @testset "Dual" begin # some weird Real subtype that we should basically leave alone
        @test ProjectTo(1.0)(Dual(1.0, 2.0)) isa Dual
        @test ProjectTo(1.0)(Dual(1, 2)) isa Dual
        @test ProjectTo(1.0 + 1im)(Dual(1.0, 2.0)) isa Complex{<:Dual}
        @test ProjectTo(1.0 + 1im)(
            Complex(Dual(1.0, 2.0), Dual(1.0, 2.0))
        ) isa Complex{<:Dual}
        @test ProjectTo(1.0)(Complex(Dual(1.0, 2.0), Dual(1.0, 2.0))) isa Dual
    end

    @testset "Base: arrays of numbers" begin
        pvec3 = ProjectTo([1, 2, 3])
        @test pvec3(1.0:3.0) === 1.0:3.0
        @test pvec3(1:3) == 1.0:3.0  # would prefer ===, map(Float64, dx) would do that, not important
        @test pvec3([1, 2, 3 + 4im]) == 1:3
        @test eltype(pvec3([1, 2, 3.0f0])) === Float64

        # reshape
        @test pvec3(reshape([1, 2, 3], 3, 1)) isa Vector
        @test_throws DimensionMismatch pvec3(reshape([1, 2, 3], 1, 3))
        @test_throws DimensionMismatch pvec3([1, 2, 3, 4])

        pmat = ProjectTo(rand(2, 2) .+ im)
        @test pmat([1 2; 3 4.0+5im]') isa Adjoint     # pass-through
        @test pmat([1 2; 3 4]') isa Matrix{ComplexF64}  # broadcast type change

        pmat2 = ProjectTo(rand(2, 2)')
        @test pmat2([1 2; 3 4.0+5im]) isa Matrix  # adjoint matrices are not re-created

        prow = ProjectTo([1im 2 3im])
        @test prow(transpose([1, 2, 3 + 4.0im])) == [1 2 3 + 4im]
        @test prow(transpose([1, 2, 3 + 4.0im])) isa Matrix  # row vectors may not pass through 
        @test prow(adjoint([1, 2, 3 + 5im])) == [1 2 3 - 5im]
        @test prow(adjoint([1, 2, 3])) isa Matrix
    end

    @testset "Base: arrays of arrays, etc" begin
        pvecvec = ProjectTo([[1, 2], [3, 4, 5]])
        @test pvecvec([1:2, 3:5])[1] == 1:2
        @test pvecvec([[1, 2 + 3im], [4 + 5im, 6, 7]])[2] == [4, 6, 7]
        @test pvecvec(hcat([1:2, hcat(3:5)]))[2] isa Vector  # reshape inner & outer

        pvecvec2 = ProjectTo(reshape(Any[[1 2], [3 4 5]], 1, 2))  # a row of rows
        y1 = pvecvec2([[1, 2], [3, 4, 5]]')
        @test y1[1] == [1 2]
        @test !(y1 isa Adjoint) && !(y1[1] isa Adjoint)

        # arrays of unknown things
        @test_throws MethodError ProjectTo([:x, :y])
        @test_throws MethodError ProjectTo(Any[:x, :y])

        @test ProjectTo(Any[1, 2])(1:2) == [1.0, 2.0]  # projects each number.
        @test Tuple(ProjectTo(Any[1, 2 + 3im])(1:2)) === (1.0, 2.0 + 0.0im)
        @test ProjectTo(Any[true, false]) isa ProjectTo{NoTangent}

        # empty arrays
        @test isempty(ProjectTo([])(1:0))
        @test_throws DimensionMismatch ProjectTo(Int[])([2])
        @test ProjectTo(Bool[]) isa ProjectTo{NoTangent}
    end

    @testset "Base: zero-arrays & Ref" begin
        pzed = ProjectTo(fill(1.0))
        @test pzed(fill(3.14)) == fill(3.14)  # easy
        @test pzed(fill(3)) == fill(3.0)      # broadcast type change must not produce number
        @test pzed(hcat(3.14)) == fill(3.14)  # reshape of length-1 array
        @test pzed(3.14 + im) == fill(3.14)   # re-wrap of a scalar number

        @test_throws DimensionMismatch ProjectTo([1])(3.14 + im) # other array projectors don't accept numbers
        @test_throws DimensionMismatch ProjectTo(hcat([1, 2]))(3.14)
        @test pzed isa ProjectTo{AbstractArray}

        pref = ProjectTo(Ref(2.0))
        @test pref(Ref(3 + im))[] === 3.0
        @test pref(4)[] === 4.0  # also re-wraps scalars
        @test pref(Ref{Any}(5.0)) isa Base.RefValue{Float64}
        pref2 = ProjectTo(Ref{Any}(6 + 7im))
        @test pref2(Ref(8))[] === 8.0 + 0.0im

        prefvec = ProjectTo(Ref([1, 2, 3 + 4im]))  # recurses into contents
        @test prefvec(Ref(1:3)) isa Base.RefValue{Vector{ComplexF64}}
        @test_throws DimensionMismatch prefvec(Ref{Any}(1:5))
    end

    #####
    ##### `LinearAlgebra`
    #####

    @testset "LinearAlgebra: $adj vectors" for adj in [transpose, adjoint]
        # adjoint vectors
        padj = ProjectTo(adj([1, 2, 3]))
        adjT = typeof(adj([1, 2, 3.0]))
        @test padj(transpose(1:3)) isa adjT
        @test padj([4 5 6 + 7im]) isa adjT
        @test padj([4.0 5.0 6.0]) isa adjT

        @test_throws DimensionMismatch padj([1, 2, 3])
        @test_throws DimensionMismatch padj([1 2 3]')
        @test_throws DimensionMismatch padj([1 2 3 4])

        padj_complex = ProjectTo(adj([1, 2, 3 + 4im]))
        @test padj_complex([4 5 6 + 7im]) == [4 5 6 + 7im]
        @test padj_complex(transpose([4, 5, 6 + 7im])) == [4 5 6 + 7im]
        @test padj_complex(adjoint([4, 5, 6 + 7im])) == [4 5 6 - 7im]

        # evil test case
        if VERSION >= v"1.7-"  # up to 1.6  Vector[[1,2,3]]'  is an error, not sure why it's called
            xs = adj(Any[Any[1, 2, 3], Any[4 + im, 5 - im, 6 + im, 7 - im]])
            pvecvec3 = ProjectTo(xs)
            @test pvecvec3(xs)[1] == [1 2 3]
            @test pvecvec3(xs)[2] == adj.([4 + im 5 - im 6 + im 7 - im])
            @test pvecvec3(xs)[2] isa LinearAlgebra.AdjOrTransAbsMat{ComplexF64,<:Vector}
            @test pvecvec3(collect(xs))[1] == [1 2 3]
            ys = permutedims([[1 2 3 + im], Any[4 5 6 7 + 8im]])
            @test pvecvec3(ys)[1] == [1 2 3]
            @test pvecvec3(ys)[2] == [4 5 6 7 + 8im]
            @test pvecvec3(xs)[2] isa LinearAlgebra.AdjOrTransAbsMat{ComplexF64,<:Vector}
            @test pvecvec3(ys) isa LinearAlgebra.AdjOrTransAbsVec

            zs = adj([[1 2; 3 4], [5 6; 7 8+im]'])
            pvecmat = ProjectTo(zs)
            @test pvecmat(zs) == zs
            @test pvecmat(collect.(zs)) == zs
            @test pvecmat(collect.(zs)) isa LinearAlgebra.AdjOrTransAbsVec
        end
    end

    @testset "LinearAlgebra: dense structured matrices" begin
        psymm = ProjectTo(Symmetric(rand(3, 3)))
        @test psymm(reshape(1:9, 3, 3)) == [1.0 3.0 5.0; 3.0 5.0 7.0; 5.0 7.0 9.0]
        @test psymm(psymm(reshape(1:9, 3, 3))) == psymm(reshape(1:9, 3, 3))
        @test psymm(rand(ComplexF32, 3, 3, 1)) isa Symmetric{Float64}
        @test ProjectTo(Symmetric(randn(3, 3) .> 0))(randn(3, 3)) == NoTangent() # Bool

        pherm = ProjectTo(Hermitian(rand(3, 3) .+ im, :L))
        # NB, projection onto Hermitian subspace, not application of Hermitian constructor
        @test pherm(reshape(1:9, 3, 3) .+ im) == [1.0 3.0 5.0; 3.0 5.0 7.0; 5.0 7.0 9.0]
        @test pherm(pherm(reshape(1:9, 3, 3))) == pherm(reshape(1:9, 3, 3))
        @test pherm(rand(ComplexF32, 3, 3, 1)) isa Hermitian{ComplexF64}

        pupp = ProjectTo(UpperTriangular(rand(3, 3)))
        @test pupp(reshape(1:9, 3, 3)) == [1.0 4.0 7.0; 0.0 5.0 8.0; 0.0 0.0 9.0]
        @test pupp(pupp(reshape(1:9, 3, 3))) == pupp(reshape(1:9, 3, 3))
        @test pupp(rand(ComplexF32, 3, 3, 1)) isa UpperTriangular{Float64}
        @test ProjectTo(UpperTriangular(randn(3, 3) .> 0))(randn(3, 3)) == NoTangent()

        # an experiment with allowing subspaces which aren't subtypes
        @test psymm(Diagonal([1, 2, 3])) isa Diagonal{Float64}
        @test pupp(Diagonal([1, 2, 3 + 4im])) isa Diagonal{Float64}
    end

    @testset "LinearAlgebra: sparse structured matrices" begin
        pdiag = ProjectTo(Diagonal(1:3))
        @test pdiag(reshape(1:9, 3, 3)) == Diagonal([1, 5, 9])
        @test pdiag(pdiag(reshape(1:9, 3, 3))) == pdiag(reshape(1:9, 3, 3))
        @test pdiag(rand(ComplexF32, 3, 3)) isa Diagonal{Float64}
        @test pdiag(Diagonal(1.0:3.0)) === Diagonal(1.0:3.0)
        @test ProjectTo(Diagonal(randn(3) .> 0))(randn(3, 3)) == NoTangent()
        @test ProjectTo(Diagonal(randn(3) .> 0))(Diagonal(rand(3))) == NoTangent()

        pbi = ProjectTo(Bidiagonal(rand(3, 3), :L))
        @test pbi(reshape(1:9, 3, 3)) == [1.0 0.0 0.0; 2.0 5.0 0.0; 0.0 6.0 9.0]
        @test pbi(pbi(reshape(1:9, 3, 3))) == pbi(reshape(1:9, 3, 3))
        @test pbi(rand(ComplexF32, 3, 3)) isa Bidiagonal{Float64}
        bi = Bidiagonal(rand(3, 3) .+ im, :L)
        @test pbi(bi) == real(bi)  # reconstruct via generic_projector
        bu = Bidiagonal(rand(3, 3) .+ im, :U)  # differs but uplo, not type
        @test pbi(bu) == diagm(0 => diag(real(bu)))
        @test_throws DimensionMismatch pbi(rand(ComplexF32, 3, 2))

        pstri = ProjectTo(SymTridiagonal(Symmetric(rand(3, 3))))
        @test pstri(reshape(1:9, 3, 3)) == [1.0 3.0 0.0; 3.0 5.0 7.0; 0.0 7.0 9.0]
        @test pstri(pstri(reshape(1:9, 3, 3))) == pstri(reshape(1:9, 3, 3))
        @test pstri(rand(ComplexF32, 3, 3)) isa SymTridiagonal{Float64}
        stri = SymTridiagonal(Symmetric(rand(3, 3) .+ im))
        @test pstri(stri) == real(stri)
        @test_throws DimensionMismatch pstri(rand(ComplexF32, 3, 2))

        ptri = ProjectTo(Tridiagonal(rand(3, 3)))
        @test ptri(reshape(1:9, 3, 3)) == [1.0 4.0 0.0; 2.0 5.0 8.0; 0.0 6.0 9.0]
        @test ptri(ptri(reshape(1:9, 3, 3))) == ptri(reshape(1:9, 3, 3))
        @test ptri(rand(ComplexF32, 3, 3)) isa Tridiagonal{Float64}
        @test_throws DimensionMismatch ptri(rand(ComplexF32, 3, 2))
    end

    #####
    ##### `SparseArrays`
    #####

    @testset "SparseArrays" begin
        # vector
        v = sprand(30, 0.3)
        pv = ProjectTo(v)

        @test pv(v) == v
        @test pv(v .* (1 + im)) ≈ v  # same nonzero elements

        o = pv(ones(Int, 30, 1))  # dense array
        @test nnz(o) == nnz(v)

        v2 = sprand(30, 0.7)  # different nonzero elements
        @test pv(v2) == pv(collect(v2))

        # matrix
        m = sprand(10, 10, 0.3)
        pm = ProjectTo(m)

        @test pm(m) == m
        @test pm(m .* (1 + im)) ≈ m
        om = pm(ones(Int, 10, 10))
        @test nnz(om) == nnz(m)

        m2 = sprand(10, 10, 0.5)
        @test pm(m2) == pm(collect(m2))

        @test_throws DimensionMismatch pv(ones(Int, 1, 30))
        @test_throws DimensionMismatch pm(ones(Int, 5, 20))
    end

    #####
    ##### `OffsetArrays`
    #####

    @testset "OffsetArrays" begin
        # While there is no code for this, the rule that it checks axes(x) == axes(dx) else
        # reshape means that it restores offsets. (It throws an error on nontrivial size mismatch.)

        poffv = ProjectTo(OffsetArray(rand(3), 0:2))
        @test axes(poffv([1, 2, 3])) == (0:2,)
        @test axes(poffv(hcat([1, 2, 3]))) == (0:2,)

        @test axes(poffv(OffsetArray(rand(3), 0:2))) == (0:2,)
        @test axes(poffv(OffsetArray(rand(3, 1), 0:2, 0:0))) == (0:2,)
    end

    #####
    ##### `ChainRulesCore`
    #####

    @testset "AbstractZero" begin
        pz = ProjectTo(ZeroTangent())
        pz(0) == NoTangent()
        @test_broken pz(ZeroTangent()) === ZeroTangent()  # not sure how NB this is to preserve
        @test pz(NoTangent()) === NoTangent()

        pb = ProjectTo(true) # Bool is categorical
        @test pb(2) === NoTangent()

        # all projectors preserve Zero, and specific type, via one fallback method:
        @test ProjectTo(pi)(ZeroTangent()) === ZeroTangent()
        @test ProjectTo(pi)(NoTangent()) === NoTangent()
        pv = ProjectTo(sprand(30, 0.3))
        @test pv(ZeroTangent()) === ZeroTangent()
        @test pv(NoTangent()) === NoTangent()
    end

    @testset "Thunk" begin
        th = @thunk 1 + 2 + 3
        pth = ProjectTo(4 + 5im)(th)
        @test pth isa Thunk
        @test unthunk(pth) === 6.0 + 0.0im
    end

    @testset "display" begin
        @test repr(ProjectTo(1.1)) == "ProjectTo{Float64}()"
        @test occursin("ProjectTo{AbstractArray}(element", repr(ProjectTo([1, 2, 3])))
        str = repr(ProjectTo([1, 2, 3]'))
        @test eval(Meta.parse(str))(ones(1, 3)) isa Adjoint{Float64,Vector{Float64}}
    end

    VERSION > v"1.1" && @testset "allocation tests" begin
        # For sure these fail on Julia 1.0, not sure about 1.3 etc.
        # We only really care about current stable anyway
        # Each "@test 33 > ..." is zero on nightly, 32 on 1.5.

        pvec = ProjectTo(rand(10^3))
        @test 0 == @ballocated $pvec(dx) setup=(dx = rand(10^3))    # pass through
        @test 90 > @ballocated $pvec(dx) setup=(dx = rand(10^3, 1))  # reshape

        @test 33 > @ballocated ProjectTo(x)(dx) setup = (x = rand(10^3); dx = rand(10^3)) # including construction

        padj = ProjectTo(adjoint(rand(10^3)))
        @test 0 == @ballocated $padj(dx) setup=(dx = adjoint(rand(10^3)))
        @test 0 == @ballocated $padj(dx) setup=(dx = transpose(rand(10^3)))

        @test 33 > @ballocated ProjectTo(x')(dx') setup = (x = rand(10^3); dx = rand(10^3))

        pdiag = ProjectTo(Diagonal(rand(10^3)))
        @test 0 == @ballocated $pdiag(dx) setup=(dx = Diagonal(rand(10^3)))

        psymm = ProjectTo(Symmetric(rand(10^3, 10^3)))
        @test_broken 0 == @ballocated $psymm(dx) setup=(dx = Symmetric(rand(10^3, 10^3)))  # 64
    end
end
