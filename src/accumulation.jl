"""
    accumulate!!(x, y)

Returns `x+y`, potentially mutating `x` in-place to hold this value.
This avoids allocations when `x` can be mutated in this way.

See also: [`InplaceableThunk`](@ref).
"""
accumulate!!(x, y) = x + y

accumulate!!(x, t::InplaceableThunk) = t.add!(x)

function accumulate!!(x::Array{<:Any, N}, y::AbstractArray{<:Any, N}) where N
    return x .+= y
end
