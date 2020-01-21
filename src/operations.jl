# TODO: This all needs a fair bit of rethinking

"""
    accumulate(Δ, ∂)

Return `Δ + ∂` evaluated in a manner that supports ChainRulesCore's
various [`AbstractDifferential`](@ref) types.

See also: [`accumulate!`](@ref), [`store!`](@ref)
"""
accumulate(Δ, ∂) = Δ + ∂

"""
    accumulate!(Δ, ∂)

Similar to [`accumulate`](@ref), but attempts to compute `Δ + rule(args...)` in-place,
storing the result in `Δ`.

!!! note
    This function may not actually store the result in `Δ` if `Δ` is immutable,
    so it is best to always call this as `Δ = accumulate!(Δ, ∂)` just in case.

This function is overloadable by using a [`InplaceThunk`](@ref).
See also: [`accumulate`](@ref), [`store!`](@ref).
"""
accumulate!(Δ, ∂) = accumulate(Δ, ∂)
accumulate!(Δ::AbstractArray, ∂) = Δ .+= ∂
