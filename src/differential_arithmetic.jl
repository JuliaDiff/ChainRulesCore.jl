#==
All differentials need to define + and *.
That happens here.

We just use @eval to define all the combinations for AbstractDifferential
subtypes, as we know the full set that might be encountered.
Thus we can avoid any ambiguities.

Notice:
    The precedence goes:
    `DoesNotExist, Zero, One, AbstractThunk, Composite, Any`
    Thus each of the @eval loops create most definitions of + and *
    defines the combination this type with all types of  lower precidence.
    This means each eval loops is 1 item smaller than the previous.
==#

Base.:+(::DoesNotExist, ::DoesNotExist) = DoesNotExist()
Base.:-(::DoesNotExist, ::DoesNotExist) = DoesNotExist()
Base.:-(::DoesNotExist) = DoesNotExist()
Base.:*(::DoesNotExist, ::DoesNotExist) = DoesNotExist()
LinearAlgebra.dot(::DoesNotExist, ::DoesNotExist) = DoesNotExist()
for T in (:One, :AbstractThunk, :Composite, :Any)
    @eval Base.:+(::DoesNotExist, b::$T) = b
    @eval Base.:+(a::$T, ::DoesNotExist) = a
    @eval Base.:-(::DoesNotExist, b::$T) = -b
    @eval Base.:-(a::$T, ::DoesNotExist) = a

    @eval Base.:*(::DoesNotExist, ::$T) = DoesNotExist()
    @eval Base.:*(::$T, ::DoesNotExist) = DoesNotExist()

    @eval LinearAlgebra.dot(::DoesNotExist, ::$T) = DoesNotExist()
    @eval LinearAlgebra.dot(::$T, ::DoesNotExist) = DoesNotExist()
end
# `DoesNotExist` and `Zero` have special relationship,
# DoesNotExist wins add, Zero wins *. This is (in theory) to allow `*` to be used for
# selecting things.
Base.:+(::DoesNotExist, ::Zero) = DoesNotExist()
Base.:+(::Zero, ::DoesNotExist) = DoesNotExist()
Base.:-(::DoesNotExist, ::Zero) = DoesNotExist()
Base.:-(::Zero, ::DoesNotExist) = DoesNotExist()
Base.:*(::DoesNotExist, ::Zero) = Zero()
Base.:*(::Zero, ::DoesNotExist) = Zero()

LinearAlgebra.dot(::DoesNotExist, ::Zero) = Zero()
LinearAlgebra.dot(::Zero, ::DoesNotExist) = Zero()

Base.muladd(::Zero, x, y) = y
Base.muladd(x, ::Zero, y) = y
Base.muladd(x, y, ::Zero) = x*y

Base.muladd(::Zero, ::Zero, y) = y
Base.muladd(x, ::Zero, ::Zero) = Zero()
Base.muladd(::Zero, x, ::Zero) = Zero()

Base.muladd(::Zero, ::Zero, ::Zero) = Zero()

Base.:+(::Zero, ::Zero) = Zero()
Base.:-(::Zero, ::Zero) = Zero()
Base.:-(::Zero) = Zero()
Base.:*(::Zero, ::Zero) = Zero()
LinearAlgebra.dot(::Zero, ::Zero) = Zero()
for T in (:One, :AbstractThunk, :Composite, :Any)
    @eval Base.:+(::Zero, b::$T) = b
    @eval Base.:+(a::$T, ::Zero) = a
    @eval Base.:-(::Zero, b::$T) = -b
    @eval Base.:-(a::$T, ::Zero) = a

    @eval Base.:*(::Zero, ::$T) = Zero()
    @eval Base.:*(::$T, ::Zero) = Zero()

    @eval LinearAlgebra.dot(::Zero, ::$T) = Zero()
    @eval LinearAlgebra.dot(::$T, ::Zero) = Zero()
end

Base.real(::Zero) = Zero()
Base.imag(::Zero) = Zero()

Base.real(::One) = One()
Base.imag(::One) = Zero()

Base.complex(::Zero) = Zero()
Base.complex(::Zero, ::Zero) = Zero()
Base.complex(::Zero, i::Real) = complex(oftype(i, 0), i)
Base.complex(r::Real, ::Zero) = complex(r)

Base.complex(::One) = One()
Base.complex(::Zero, ::One) = im
Base.complex(::One, ::Zero) = One()

Base.:+(a::One, b::One) = extern(a) + extern(b)
Base.:*(::One, ::One) = One()
for T in (:AbstractThunk, :Composite, :Any)
    if T != :Composite
        @eval Base.:+(a::One, b::$T) = extern(a) + b
        @eval Base.:+(a::$T, b::One) = a + extern(b)
    end

    @eval Base.:*(::One, b::$T) = b
    @eval Base.:*(a::$T, ::One) = a
end

LinearAlgebra.dot(::One, x::Number) = x
LinearAlgebra.dot(x::Number, ::One) = conj(x)  # see definition of Frobenius inner product

Base.:+(a::AbstractThunk, b::AbstractThunk) = unthunk(a) + unthunk(b)
Base.:*(a::AbstractThunk, b::AbstractThunk) = unthunk(a) * unthunk(b)
for T in (:Composite, :Any)
    @eval Base.:+(a::AbstractThunk, b::$T) = unthunk(a) + b
    @eval Base.:+(a::$T, b::AbstractThunk) = a + unthunk(b)

    @eval Base.:*(a::AbstractThunk, b::$T) = unthunk(a) * b
    @eval Base.:*(a::$T, b::AbstractThunk) = a * unthunk(b)
end

function Base.:+(a::Composite{P}, b::Composite{P}) where P
    data = elementwise_add(backing(a), backing(b))
    return Composite{P, typeof(data)}(data)
end
function Base.:+(a::P, d::Composite{P}) where P
    net_backing = elementwise_add(backing(a), backing(d))
    if debug_mode()
        try
            return construct(P, net_backing)
        catch err
            throw(PrimalAdditionFailedException(a, d, err))
        end
    else
        return construct(P, net_backing)
    end
end
Base.:+(a::Dict, d::Composite{P}) where {P} = merge(+, a, backing(d))
Base.:+(a::Composite{P}, b::P) where P = b + a

# We intentionally do not define, `Base.*(::Composite, ::Composite)` as that is not meaningful
# In general one doesn't have to represent multiplications of 2 differentials
# Only of a differential and a scaling factor (generally `Real`)
for T in (:Any,)
    @eval Base.:*(s::$T, comp::Composite) = map(x->s*x, comp)
    @eval Base.:*(comp::Composite, s::$T) = map(x->x*s, comp)
end
