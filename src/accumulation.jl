"""
    add!!(x, y)

Returns `x+y`, potentially mutating `x` in-place to hold this value.
This avoids allocations when `x` can be mutated in this way.

See also: [`InplaceableThunk`](@ref).
"""
add!!(x, y) = x + y

add!!(x, t::InplaceableThunk) = t.add!(x)

function add!!(x::Array{<:Any, N}, y::AbstractArray{<:Any, N}) where N
    return x .+= y
end
