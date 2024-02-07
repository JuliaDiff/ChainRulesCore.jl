#==
All tangents need to define + and *.
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
Base.:+(x::NotImplemented, ::NotImplemented) = x
Base.:*(x::NotImplemented, ::NotImplemented) = x
LinearAlgebra.dot(x::NotImplemented, ::NotImplemented) = x
# `NotImplemented` always "wins" +
for T in (:ZeroTangent, :NoTangent, :AbstractThunk, :StructuralTangent, :Any)
    @eval Base.:+(x::NotImplemented, ::$T) = x
    @eval Base.:+(::$T, x::NotImplemented) = x
end
# `NotImplemented` "loses" * and dot against NoTangent and ZeroTangent
# this can be used to ignore partial derivatives that are not implemented
for T in (:ZeroTangent, :NoTangent)
    @eval Base.:*(::NotImplemented, ::$T) = $T()
    @eval Base.:*(::$T, ::NotImplemented) = $T()
    @eval LinearAlgebra.dot(::NotImplemented, ::$T) = $T()
    @eval LinearAlgebra.dot(::$T, ::NotImplemented) = $T()
end
# `NotImplemented` "wins" * and dot for other types
for T in (:AbstractThunk, :StructuralTangent, :Any)
    @eval Base.:*(x::NotImplemented, ::$T) = x
    @eval Base.:*(::$T, x::NotImplemented) = x
    @eval LinearAlgebra.dot(x::NotImplemented, ::$T) = x
    @eval LinearAlgebra.dot(::$T, x::NotImplemented) = x
end
# unary :- is the same as multiplication by -1
Base.:-(x::NotImplemented) = x

# subtraction throws an exception: in AD we add tangents but do not subtract them
# subtraction happens eg. in gradient descent which can't be performed with `NotImplemented`
Base.:-(x::NotImplemented, ::NotImplemented) = throw(NotImplementedException(x))
for T in (:ZeroTangent, :NoTangent, :AbstractThunk, :Tangent, :Any)
    @eval Base.:-(x::NotImplemented, ::$T) = throw(NotImplementedException(x))
    @eval Base.:-(::$T, x::NotImplemented) = throw(NotImplementedException(x))
end

Base.:+(::NoTangent, ::NoTangent) = NoTangent()
Base.:-(::NoTangent, ::NoTangent) = NoTangent()
Base.:-(::NoTangent) = NoTangent()
Base.:*(::NoTangent, ::NoTangent) = NoTangent()
LinearAlgebra.dot(::NoTangent, ::NoTangent) = NoTangent()
for T in (:AbstractThunk, :StructuralTangent, :Any)
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
Base.muladd(x, y, ::ZeroTangent) = x * y

Base.muladd(::ZeroTangent, ::ZeroTangent, y) = y
Base.muladd(x, ::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.muladd(::ZeroTangent, x, ::ZeroTangent) = ZeroTangent()

Base.muladd(::ZeroTangent, ::ZeroTangent, ::ZeroTangent) = ZeroTangent()

Base.:+(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.:-(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
Base.:-(::ZeroTangent) = ZeroTangent()
Base.:*(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
LinearAlgebra.dot(::ZeroTangent, ::ZeroTangent) = ZeroTangent()
for T in (:AbstractThunk, :StructuralTangent, :Any)
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

function Base.:+(a::StructuralTangent{P}, b::StructuralTangent{P}) where {P}
    data = elementwise_add(backing(a), backing(b))
    return StructuralTangent{P}(data)
end
function Base.:+(a::P, d::StructuralTangent{P}) where {P}
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
Base.:+(a::StructuralTangent{P}, b::P) where {P} = b + a

Base.:-(tangent::StructuralTangent{P}) where {P} = map(-, tangent)
Base.:-(a::StructuralTangent{P}, b::StructuralTangent{P}) where {P} = a + (-b)

# We intentionally do not define, `Base.*(::Tangent, ::Tangent)` as that is not meaningful
# In general one doesn't have to represent multiplications of 2 tangents
# Only of a tangent and a scaling factor (generally `Real`)
for T in (:Number,)
    @eval Base.:*(s::$T, tangent::StructuralTangent) = map(x -> s * x, tangent)
    @eval Base.:*(tangent::StructuralTangent, s::$T) = map(x -> x * s, tangent)
end
