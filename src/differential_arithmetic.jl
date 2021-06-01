#==
All differentials need to define + and *.
That happens here.

We just use @eval to define all the combinations for AbstractTangent
subtypes, as we know the full set that might be encountered.
Thus we can avoid any ambiguities.

Notice:
    The precedence goes:
    `NotImplemented, NoTangent, ZeroTangent, AbstractThunk, Tangent, Any`
    Thus each of the @eval loops create most definitions of + and *
    defines the combination this type with all types of  lower precidence.
    This means each eval loops is 1 item smaller than the previous.
==#

# we propagate `NotImplemented` (e.g., in `@scalar_rule`)
# this requires the following definitions (see also #337)
Base.:+(x::NotImplemented, ::ZeroTangent) = x
Base.:+(::ZeroTangent, x::NotImplemented) = x
Base.:+(x::NotImplemented, ::NotImplemented) = x
Base.:*(::NotImplemented, ::ZeroTangent) = ZeroTangent()
Base.:*(::ZeroTangent, ::NotImplemented) = ZeroTangent()
for T in (:NoTangent, :AbstractThunk, :Tangent, :Any)
    @eval Base.:+(x::NotImplemented, ::$T) = x
    @eval Base.:+(::$T, x::NotImplemented) = x
    @eval Base.:*(x::NotImplemented, ::$T) = x
end
Base.muladd(x::NotImplemented, y, z) = x
Base.muladd(::NotImplemented, ::ZeroTangent, z) = z
Base.muladd(x::NotImplemented, y, ::ZeroTangent) = x
Base.muladd(::NotImplemented, ::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.muladd(x, y::NotImplemented, z) = y
Base.muladd(::ZeroTangent, ::NotImplemented, z) = z
Base.muladd(x, y::NotImplemented, ::ZeroTangent) = y
Base.muladd(::ZeroTangent, ::NotImplemented, ::ZeroTangent) = ZeroTangent()
Base.muladd(x, y, z::NotImplemented) = z
Base.muladd(::ZeroTangent, y, z::NotImplemented) = z
Base.muladd(x, ::ZeroTangent, z::NotImplemented) = z
Base.muladd(::ZeroTangent, ::ZeroTangent, z::NotImplemented) = z
Base.muladd(x::NotImplemented, ::NotImplemented, z) = x
Base.muladd(x::NotImplemented, ::NotImplemented, ::ZeroTangent) = x
Base.muladd(x::NotImplemented, y, ::NotImplemented) = x
Base.muladd(::NotImplemented, ::ZeroTangent, z::NotImplemented) = z
Base.muladd(x, y::NotImplemented, ::NotImplemented) = y
Base.muladd(::ZeroTangent, ::NotImplemented, z::NotImplemented) = z
Base.muladd(x::NotImplemented, ::NotImplemented, ::NotImplemented) = x
LinearAlgebra.dot(::NotImplemented, ::ZeroTangent) = ZeroTangent()
LinearAlgebra.dot(::ZeroTangent, ::NotImplemented) = ZeroTangent()

# other common operations throw an exception
Base.:+(x::NotImplemented) = throw(NotImplementedException(x))
Base.:-(x::NotImplemented) = throw(NotImplementedException(x))
Base.:-(x::NotImplemented, ::ZeroTangent) = throw(NotImplementedException(x))
Base.:-(::ZeroTangent, x::NotImplemented) = throw(NotImplementedException(x))
Base.:-(x::NotImplemented, ::NotImplemented) = throw(NotImplementedException(x))
Base.:*(x::NotImplemented, ::NotImplemented) = throw(NotImplementedException(x))
function LinearAlgebra.dot(x::NotImplemented, ::NotImplemented)
    return throw(NotImplementedException(x))
end
for T in (:NoTangent, :AbstractThunk, :Tangent, :Any)
    @eval Base.:-(x::NotImplemented, ::$T) = throw(NotImplementedException(x))
    @eval Base.:-(::$T, x::NotImplemented) = throw(NotImplementedException(x))
    @eval Base.:*(::$T, x::NotImplemented) = throw(NotImplementedException(x))
    @eval LinearAlgebra.dot(x::NotImplemented, ::$T) = throw(NotImplementedException(x))
    @eval LinearAlgebra.dot(::$T, x::NotImplemented) = throw(NotImplementedException(x))
end

Base.:+(::NoTangent, ::NoTangent) = NoTangent()
Base.:-(::NoTangent, ::NoTangent) = NoTangent()
Base.:-(::NoTangent) = NoTangent()
Base.:*(::NoTangent, ::NoTangent) = NoTangent()
LinearAlgebra.dot(::NoTangent, ::NoTangent) = NoTangent()
for T in (:AbstractThunk, :Tangent, :Any)
    @eval Base.:+(::NoTangent, b::$T) = b
    @eval Base.:+(a::$T, ::NoTangent) = a
    @eval Base.:-(::NoTangent, b::$T) = -b
    @eval Base.:-(a::$T, ::NoTangent) = a

    @eval Base.:*(::NoTangent, ::$T) = NoTangent()
    @eval Base.:*(::$T, ::NoTangent) = NoTangent()

    @eval LinearAlgebra.dot(::NoTangent, ::$T) = NoTangent()
    @eval LinearAlgebra.dot(::$T, ::NoTangent) = NoTangent()
end
# `NoTangent` and `ZeroTangent` have special relationship,
# NoTangent wins add, ZeroTangent wins *. This is (in theory) to allow `*` to be used for
# selecting things.
Base.:+(::NoTangent, ::ZeroTangent) = NoTangent()
Base.:+(::ZeroTangent, ::NoTangent) = NoTangent()
Base.:-(::NoTangent, ::ZeroTangent) = NoTangent()
Base.:-(::ZeroTangent, ::NoTangent) = NoTangent()
Base.:*(::NoTangent, ::ZeroTangent) = ZeroTangent()
Base.:*(::ZeroTangent, ::NoTangent) = ZeroTangent()

LinearAlgebra.dot(::NoTangent, ::ZeroTangent) = ZeroTangent()
LinearAlgebra.dot(::ZeroTangent, ::NoTangent) = ZeroTangent()

Base.muladd(::ZeroTangent, x, y) = y
Base.muladd(x, ::ZeroTangent, y) = y
Base.muladd(x, y, ::ZeroTangent) = x*y

Base.muladd(::ZeroTangent, ::ZeroTangent, y) = y
Base.muladd(x, ::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.muladd(::ZeroTangent, x, ::ZeroTangent) = ZeroTangent()

Base.muladd(::ZeroTangent, ::ZeroTangent, ::ZeroTangent) = ZeroTangent()

Base.:+(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.:-(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.:-(::ZeroTangent) = ZeroTangent()
Base.:*(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
LinearAlgebra.dot(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
for T in (:AbstractThunk, :Tangent, :Any)
    @eval Base.:+(::ZeroTangent, b::$T) = b
    @eval Base.:+(a::$T, ::ZeroTangent) = a
    @eval Base.:-(::ZeroTangent, b::$T) = -b
    @eval Base.:-(a::$T, ::ZeroTangent) = a

    @eval Base.:*(::ZeroTangent, ::$T) = ZeroTangent()
    @eval Base.:*(::$T, ::ZeroTangent) = ZeroTangent()

    @eval LinearAlgebra.dot(::ZeroTangent, ::$T) = ZeroTangent()
    @eval LinearAlgebra.dot(::$T, ::ZeroTangent) = ZeroTangent()
end

Base.real(::ZeroTangent) = ZeroTangent()
Base.imag(::ZeroTangent) = ZeroTangent()

Base.complex(::ZeroTangent) = ZeroTangent()
Base.complex(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.complex(::ZeroTangent, i::Real) = complex(oftype(i, 0), i)
Base.complex(r::Real, ::ZeroTangent) = complex(r)

Base.:+(a::AbstractThunk, b::AbstractThunk) = unthunk(a) + unthunk(b)
Base.:*(a::AbstractThunk, b::AbstractThunk) = unthunk(a) * unthunk(b)
for T in (:Tangent, :Any)
    @eval Base.:+(a::AbstractThunk, b::$T) = unthunk(a) + b
    @eval Base.:+(a::$T, b::AbstractThunk) = a + unthunk(b)

    @eval Base.:*(a::AbstractThunk, b::$T) = unthunk(a) * b
    @eval Base.:*(a::$T, b::AbstractThunk) = a * unthunk(b)
end

function Base.:+(a::Tangent{P}, b::Tangent{P}) where P
    data = elementwise_add(backing(a), backing(b))
    return Tangent{P, typeof(data)}(data)
end
function Base.:+(a::P, d::Tangent{P}) where P
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
Base.:+(a::Dict, d::Tangent{P}) where {P} = merge(+, a, backing(d))
Base.:+(a::Tangent{P}, b::P) where P = b + a

# We intentionally do not define, `Base.*(::Tangent, ::Tangent)` as that is not meaningful
# In general one doesn't have to represent multiplications of 2 differentials
# Only of a differential and a scaling factor (generally `Real`)
for T in (:Any,)
    @eval Base.:*(s::$T, comp::Tangent) = map(x->s*x, comp)
    @eval Base.:*(comp::Tangent, s::$T) = map(x->x*s, comp)
end
