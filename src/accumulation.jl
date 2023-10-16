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
For arrays this means `x .= y` will mutate `x`, if `y` is an appropriate tangent.

Here "appropriate" means that `y` cannot be complex unless `x` is too,
and that for structured matrices like `x isa Diagonal`, `y` shares this structure.

!!! note "history"
    Wrapper array types should overload this function if they can be written into.
    Before ChainRulesCore 1.16, it would guess `true` for most wrappers based on `parent`,
    but this is not safe, e.g. it will lead to an error with ReadOnlyArrays.jl. 

There must always be a correct non-mutating path, so in uncertain cases,
this function returns `false`.
"""
is_inplaceable_destination(::Any) = false

is_inplaceable_destination(::Array) = true
is_inplaceable_destination(:: Array{<:Integer}) = false

function is_inplaceable_destination(x::SubArray)
    alpha = is_inplaceable_destination(parent(x))
    beta = x.indices isa Tuple{Vararg{Union{Integer, Base.Slice, UnitRange}}}
    return alpha && beta
end

for T in [:PermutedDimsArray, :ReshapedArray]
    @eval is_inplaceable_destination(x::Base.$T) = is_inplaceable_destination(parent(x))
end
for T in [:Adjoint, :Transpose, :Diagonal, :UpperTriangular, :LowerTriangular]
    @eval is_inplaceable_destination(x::LinearAlgebra.$T) = is_inplaceable_destination(parent(x))
end
# Hermitian and Symmetric are too fussy to deal with right now
# https://github.com/JuliaLang/julia/issues/38056

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
