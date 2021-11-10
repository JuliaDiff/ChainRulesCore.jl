"""
    add!!(x, y)

Returns `x+y`, potentially mutating `x` in-place to hold this value.
This avoids allocations when `x` can be mutated in this way.
"""
add!!(x, y) = x + y

"""
    add!!(x, t::InplacableThunk)

The specialization of `add!!` for [`InplaceableThunk`](@ref) promises to only call
`t.add!` on `x` if `x` is suitably mutable; otherwise it will be out of place.
"""
function add!!(x, t::InplaceableThunk)
    return if is_inplaceable_destination(x)
        if !debug_mode()
            t.add!(x)
        else
            debug_add!(x, t)
        end
    else
        x + t
    end
end

add!!(x::AbstractArray, y::Thunk) = add!!(x, unthunk(y))

function add!!(x::AbstractArray{<:Any,N}, y::AbstractArray{<:Any,N}) where {N}
    return if is_inplaceable_destination(x)
        x .+= y
    else
        x + y
    end
end

"""
    is_inplaceable_destination(x) -> Bool

Returns true if `x` is suitable for for storing inplace accumulation of gradients.
For arrays this boils down `x .= y` if will work to mutate `x`, if `y` is an appropriate
tangent.
Wrapper array types do not need to overload this if they overload `Base.parent`, and are
`is_inplaceable_destination` if and only if their parent array is.
Other types should overload this, as it defaults to `false`.
"""
is_inplaceable_destination(::Any) = false
is_inplaceable_destination(::Array) = true
is_inplaceable_destination(::SparseVector) = true
is_inplaceable_destination(::SparseMatrixCSC) = true
is_inplaceable_destination(::BitArray) = true
function is_inplaceable_destination(x::AbstractArray)
    p = parent(x)
    p === x && return false  # no parent
    # basically all wrapper types delegate `setindex!` to their `parent` after some
    # processing and so are mutable if their `parent` is.
    return is_inplaceable_destination(p)
end

# Hermitian and Symmetric are too fussy to deal with right now
# https://github.com/JuliaLang/julia/issues/38056
# TODO: https://github.com/JuliaDiff/ChainRulesCore.jl/issues/236
is_inplaceable_destination(::LinearAlgebra.Hermitian) = false
is_inplaceable_destination(::LinearAlgebra.Symmetric) = false

function debug_add!(accumuland, t::InplaceableThunk)
    returned_value = t.add!(accumuland)
    if returned_value !== accumuland
        throw(BadInplaceException(t, accumuland, returned_value))
    end
    return returned_value
end

struct BadInplaceException <: Exception
    ithunk::InplaceableThunk
    accumuland
    returned_value
end

function Base.showerror(io::IO, err::BadInplaceException)
    println(io, "`add!!(accumuland, ithunk))` did not return an updated accumuland.")
    println(io, "ithunk = $(err.ithunk)")
    println(io, "accumuland = $(err.accumuland)")
    println(io, "returned_value = $(err.returned_value)")

    if err.accumuland == err.returned_value
        println(
            io,
            "Which in this case happenned to be equal. But they are not the same object.",
        )
    end
end
