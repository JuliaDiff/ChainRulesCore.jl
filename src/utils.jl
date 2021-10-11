"""
    realdot(x, y)

Compute `real(dot(x, y))` while avoiding computing the imaginary part if possible.

This function can be useful if you implement a `rrule` for a non-holomorphic function
on complex numbers.

See also: [`imagdot`](@ref)
"""
@inline realdot(x, y) = real(dot(x, y))
@inline realdot(x::Number, y::Number) = muladd(real(x), real(y), imag(x) * imag(y))
@inline realdot(x::Real, y::Number) = x * real(y)
@inline realdot(x::Number, y::Real) = real(x) * y
@inline realdot(x::Real, y::Real) = x * y

"""
    imagdot(x, y)

Compute `imag(dot(x, y))` while avoiding computing the real part if possible.

This function can be useful if you implement a `rrule` for a non-holomorphic function
on complex numbers.

See also: [`realdot`](@ref)
"""
@inline imagdot(x, y) = imag(dot(x, y))
@inline function imagdot(x::Number, y::Number)
    return muladd(-imag(x), real(y), real(x) * imag(y))
end
@inline imagdot(x::Real, y::Number) = x * imag(y)
@inline imagdot(x::Number, y::Real) = -imag(x) * y
@inline imagdot(x::Real, y::Real) = ZeroTangent()
@inline imagdot(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}) = ZeroTangent()
