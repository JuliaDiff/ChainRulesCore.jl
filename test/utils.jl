# struct need to be defined outside of tests for julia 1.0 compat
# custom complex number to test fallback definition
struct CustomComplex{T}
    re::T
    im::T
end

Base.real(x::CustomComplex) = x.re
Base.imag(x::CustomComplex) = x.im

Base.conj(x::CustomComplex) = CustomComplex(x.re, -x.im)

Base.:*(a::CustomComplex, b::Number) = CustomComplex(reim((a.re + a.im * im) * b)...)
Base.:*(a::Number, b::CustomComplex) = b * a
function Base.:*(a::CustomComplex, b::CustomComplex)
    return CustomComplex(reim((a.re + a.im * im) * (b.re + b.im * im))...)
end

@testset "utils.jl" begin
    @testset "conjtimes" begin
        inputs = (randn(), randn(ComplexF64), CustomComplex(reim(randn(ComplexF64))...))
        for x in inputs, y in inputs
            @test realconjtimes(x, y) == real(conj(x) * y)

            if x isa Real && y isa Real
                @test imagconjtimes(x, y) === ZeroTangent()
            else
                @test imagconjtimes(x, y) == imag(conj(x) * y)
            end
        end
    end
end
