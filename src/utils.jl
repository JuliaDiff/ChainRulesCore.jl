"""
    realconjtimes(x, y)

Compute `real(conj(x) * y)` while avoiding computing the imaginary part if possible.

This function can be useful if you implement a `rrule` for a non-holomorphic function
on complex numbers.

See also: [`imagconjtimes`](@ref)
"""
@inline realconjtimes(x, y) = real(conj(x) * y)
@inline realconjtimes(x::Complex, y::Complex) = muladd(real(x), real(y), imag(x) * imag(y))
@inline realconjtimes(x::Real, y::Complex) = x * real(y)
@inline realconjtimes(x::Complex, y::Real) = real(x) * y
@inline realconjtimes(x::Real, y::Real) = x * y

"""
    imagconjtimes(x, y)

Compute `imag(conj(x) * y)` while avoiding computing the real part if possible.

This function can be useful if you implement a `rrule` for a non-holomorphic function
on complex numbers.

See also: [`realconjtimes`](@ref)
"""
@inline imagconjtimes(x, y) = imag(conj(x) * y)
@inline function imagconjtimes(x::Complex, y::Complex)
    return muladd(-imag(x), real(y), real(x) * imag(y))
end
@inline imagconjtimes(x::Real, y::Complex) = x * imag(y)
@inline imagconjtimes(x::Complex, y::Real) = -imag(x) * y
@inline imagconjtimes(x::Real, y::Real) = ZeroTangent()
