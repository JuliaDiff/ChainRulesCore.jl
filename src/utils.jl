"""
    realdot(x, y)

Compute `real(dot(x, y))` while avoiding computing the imaginary part if possible.

This function can be useful if you implement a `rrule` for a non-holomorphic function
on complex numbers.

See also: [`imagdot`](@ref)
"""
@inline realdot(x, y) = real(dot(x, y))
@inline realdot(x::Complex, y::Complex) = muladd(real(x), real(y), imag(x) * imag(y))
@inline realdot(x::Real, y::Complex) = x * real(y)
@inline realdot(x::Complex, y::Real) = real(x) * y
@inline realdot(x::Real, y::Real) = x * y
@inline realdot(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = dot(x, y)

"""
    imagdot(x, y)

Compute `imag(dot(x, y))` while avoiding computing the real part if possible.

This function can be useful if you implement a `rrule` for a non-holomorphic function
on complex numbers.

See also: [`realdot`](@ref)
"""
@inline imagdot(x, y) = imag(dot(x, y))
@inline function imagdot(x::Complex, y::Complex)
    return muladd(-imag(x), real(y), real(x) * imag(y))
end
@inline imagdot(x::Real, y::Complex) = x * imag(y)
@inline imagdot(x::Complex, y::Real) = -imag(x) * y
@inline imagdot(x::Real, y::Real) = ZeroTangent()
@inline imagdot(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = ZeroTangent()
