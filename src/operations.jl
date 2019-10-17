# TODO: This all needs a fair bit of rethinking

"""
    accumulate(Δ, ∂)

Return `Δ + ∂` evaluated in a manner that supports ChainRulesCore's
various `AbstractDifferential` types.

See also: [`accumulate!`](@ref), [`store!`](@ref), [`AbstractRule`](@ref)
"""
accumulate(Δ, ∂) = Δ .+ ∂

"""
    accumulate!(Δ, ∂)

Similar to [`accumulate`](@ref), but attempts to compute `Δ + rule(args...)` in-place,
storing the result in `Δ`.

Note: this function may not actually store the result in `Δ` if `Δ` is immutable,
so it is best to always call this as `Δ = accumulate!(Δ, ∂)` just in-case.

This function is overloadable by using a [`InplaceThunk`](@ref).
See also: [`accumulate`](@ref), [`store!`](@ref).
"""
accumulate!(Δ, ∂) = store!(Δ, accumulate(Δ, ∂))

accumulate!(Δ::Number, ∂) = accumulate(Δ, ∂)


"""
    store!(Δ, ∂)

Stores `∂`, in `Δ`, overwriting what ever was in `Δ` before.
potentially avoiding intermediate temporary allocations that might be
necessary for alternative approaches  (e.g. `copyto!(Δ, extern(∂))`)

Like [`accumulate`](@ref) and [`accumulate!`](@ref), this function is intended
to be customizable for specific rules/input types.

See also: [`accumulate`](@ref), [`accumulate!`](@ref), [`AbstractRule`](@ref)
"""
store!(Δ, ∂) = materialize!(Δ, broadcastable(∂))
