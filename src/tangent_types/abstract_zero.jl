"""
    AbstractZero <: AbstractTangent

Supertype for zero-like tangents—i.e., tangents that act like zero when
added or multiplied to other values.
If an AD system encounters a propagator that takes as input only subtypes of `AbstractZero`,
then it can stop performing AD operations.
All propagators are linear functions, and thus the final result will be zero.

All `AbstractZero` subtypes are singleton types.
There are two of them: [`ZeroTangent()`](@ref) and [`NoTangent()`](@ref).
"""
abstract type AbstractZero <: AbstractTangent end
Base.iszero(::AbstractZero) = true

Base.iterate(x::AbstractZero) = (x, nothing)
Base.iterate(::AbstractZero, ::Any) = nothing

Base.Broadcast.broadcastable(x::AbstractZero) = Ref(x)
Base.Broadcast.broadcasted(::Type{T}) where {T<:AbstractZero} = T()

Base.:/(z::AbstractZero, ::Any) = z

Base.convert(::Type{T}, x::AbstractZero) where {T<:Number} = zero(T)
# (::Type{T})(::AbstractZero, ::AbstractZero...) where {T<:Number} = zero(T)

(::Type{Complex})(x::AbstractZero, y::Real) = Complex(false, y)
(::Type{Complex})(x::Real, y::AbstractZero) = Complex(x, false)

Base.getindex(z::AbstractZero, ind...) = z
Base.view(z::AbstractZero, ind...) = z
Base.sum(z::AbstractZero; dims=:) = z
Base.reshape(z::AbstractZero, size...) = z
Base.reverse(z::AbstractZero, args...; kwargs...) = z

# LinearAlgebra
LinearAlgebra.adjoint(z::AbstractZero, ind...) = z
LinearAlgebra.transpose(z::AbstractZero, ind...) = z

for T in (
        :UniformScaling, :Adjoint, :Transpose, :Diagonal
        :UpperTriangular, :LowerTriangular, :UpperHessenberg,
        :UnitUpperTriangular, :UnitLowerTriangular,
    )
    VERSION < v"1.4" && f == :UpperHessenberg && continue  # not defined in 1.0
    @eval (::Type{<:LinearAlgebra.$T})(z::AbstractZero) = z
end

LinearAlgebra.Symmetric(z::AbstractZero, uplo=:U) = z
LinearAlgebra.Hermitian(z::AbstractZero, uplo=:U) = z

LinearAlgebra.Bidiagonal(dv::AbstractVector, ev::AbstractZero, uplo::Symbol) = Diagonal(dv)
function LinearAlgebra.Bidiagonal(dv::AbstractZero, ev::AbstractVector, uplo::Symbol)
    dv = fill!(similar(ev, length(ev) + 1), 0) # can't avoid making a dummy array
    return Bidiagonal(dv, convert(typeof(dv), ev), uplo)
end
LinearAlgebra.Bidiagonal(dv::AbstractZero, ev::AbstractZero, uplo::Symbol) = NoTangent()

# one Zero:
LinearAlgebra.Tridiagonal(dl::AbstractZero, d::AbstractVector, du::AbstractVector) = Bidiagonal(_promote_vectors(d, du)..., :U)
LinearAlgebra.Tridiagonal(dl::AbstractVector, d::AbstractVector, du::AbstractZero) = Bidiagonal(_promote_vectors(d, dl)..., :L)
function LinearAlgebra.Tridiagonal(dl::AbstractVector, d::AbstractZero, du::AbstractVector)
    d = fill!(similar(dl, length(dl) + 1), 0)
    return Tridiagonal(convert(typeof(d), dl), d, convert(typeof(d), du))
end
# two Zeros:
LinearAlgebra.Tridiagonal(dl::AbstractZero, d::AbstractVector, du::AbstractZero) = Diagonal(d)
LinearAlgebra.Tridiagonal(dl::AbstractZero, d::AbstractZero, du::AbstractVector) = Bidiagonal(d, du, :U)
LinearAlgebra.Tridiagonal(dl::AbstractVector, d::AbstractZero, du::AbstractZero) = Bidiagonal(d, dl, :L)
# three Zeros:
LinearAlgebra.Tridiagonal(dl::AbstractZero, d::AbstractZero, du::AbstractZero) = NoTangent()

LinearAlgebra.SymTridiagonal(dv::AbstractVector, ev::AbstractZero) = Diagonal(dv)
function LinearAlgebra.SymTridiagonal(dv::AbstractZero, ev::AbstractVector)
    dv = fill!(similar(ev, length(ev) + 1), 0)
    return SymTridiagonal(dv, convert(typeof(dv), ev))
end
LinearAlgebra.SymTridiagonal(dv::AbstractZero, ev::AbstractZero) = NoTangent()

# These types all demand exactly same-type vectors, but may get e.g. Fill, Vector.
_promote_vectors(x::T, y::T) where {T<:AbstractVector} = (x, y)
function _promote_vectors(x::AbstractVector, y::AbstractVector)
    T = Base._return_type(+, Tuple{typeof(x), typeof(y)})
    if isconcretetype(T)
        return convert(T, x), convert(T, y)
    else
        short = map(Base.splat(first ∘ promote), zip(x, y))
        return convert(typeof(short), x), convert(typeof(short), y)
    end
end

"""
    ZeroTangent() <: AbstractZero

The additive identity for tangents.
This is basically the same as `0`.
A derivative of `ZeroTangent()` does not propagate through the primal function.
"""
struct ZeroTangent <: AbstractZero end

Base.eltype(::Type{ZeroTangent}) = ZeroTangent

Base.zero(::AbstractTangent) = ZeroTangent()
Base.zero(::Type{<:AbstractTangent}) = ZeroTangent()

"""
    NoTangent() <: AbstractZero

This tangent indicates that the derivative does not exist.
It is the tangent type for primal types that are not differentiable,
such as integers or booleans (when they are not being used to represent
floating-point values).
The only valid way to perturb such values is to not change them at all.
As a consequence, `NoTangent` is functionally identical to `ZeroTangent()`,
but it provides additional semantic information.

Adding `NoTangent()` to a primal is generally wrong: gradient-based
methods cannot be used to optimize over discrete variables.
An optimization package making use of this might want to check for such a case.

!!! note
    This does not indicate that the derivative is not implemented,
    but rather that mathematically it is not defined.

This mostly shows up as the derivative with respect to dimension, index, or size
arguments.
```
    function rrule(fill, x, len::Int)
        y = fill(x, len)
        fill_pullback(ȳ) = (NoTangent(), @thunk(sum(Ȳ)), NoTangent())
        return y, fill_pullback
    end
```
"""
struct NoTangent <: AbstractZero end
