"""
    add!!(x, y)

Returns `x+y`, potentially mutating `x` in-place to hold this value.
This avoids allocations when `x` can be mutated in this way.
"""
add!!(x, y) = x + y

"""
    add!!(x, t::ImplacableThunk)

The specialization of `add!!` for [`InplaceableThunk`](@ref) promises to only call
`t.add!` on `x` if `x` is suitably mutable; otherwise it will be out of place.
"""
function add!!(x, t::InplaceableThunk)
    return if is_inplaceable_destination(x)
        t.add!(x)
    else
        x + t
    end
end

function add!!(x::AbstractArray{<:Any, N}, y::AbstractArray{<:Any, N}) where N
    return if is_inplaceable_destination(x)
        x .+= y
    else
        x + y
    end
end


"""
    is_inplaceable_destination(x)

Returns true if `x` is suitable for for storing inplace accumulation of gradients.
For arrays this boils down `x .= y` if will work to mutate `x`, if `y` is an appropriate
differential.
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
