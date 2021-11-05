# [`ProjectTo` the primal subspace](@id projectto)

Rules with abstractly-typed arguments may return incorrect answers when called with certain concrete types.
A classic example is the matrix-matrix multiplication rule, a naive definition of which follows:
```julia
function rrule(::typeof(*), A::AbstractMatrix, B::AbstractMatrix)
    function times_pullback(ȳ)
        dA = ȳ * B'
        dB = A' * ȳ
        return NoTangent(), dA, dB
    end
    return A * B, times_pullback
end
```
When computing `*(A, B)`, where `A isa Diagonal` and `B isa Matrix`, the output will be a `Matrix`.
As a result, `ȳ` in the pullback will be a `Matrix`, and consequently `dA` for a `A isa Diagonal` will be a `Matrix`, which is wrong.
Not only is it the wrong type, but it can contain non-zeros off the diagonal, which is not possible, it is outside of the subspace.
While a specialised rules can indeed be written for the `Diagonal` case, there are many other types and we don't want to be forced to write a rule for each of them.
Instead, `project_A = ProjectTo(A)` can be used (outside the pullback) to extract an object that knows how to project onto the type of `A` (e.g. also knows the size of the array).
This object can be called with a tangent `ȳ * B'`, by doing `project_A(ȳ * B')`, to project it on the tangent space of `A`.
The correct rule then looks like
```julia
function rrule(::typeof(*), A::AbstractMatrix, B::AbstractMatrix)
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function times_pullback(ȳ)
        dA = ȳ * B'
        dB = A' * ȳ
        return NoTangent(), project_A(dA), project_B(dB)
    end
    return A * B, times_pullback
end
```

!!! note "It is often good to `@thunk` your projections"
    The above example is potentially a good place for using a [`@thunk`](@ref).
    This is not required, but can in some cases be more computationally efficient, see [Use `Thunk`s appropriately](@ref).
    When combining thunks and projections, `@thunk()` must be the outermost call.

    A more optimized implementation of the matrix-matrix multiplication example would have
    ```julia
    times_pullback(ȳ) = NoTangent(), @thunk(project_A(ȳ * B')), @thunk(project_B(A' * ȳ))
    ```
    within the `rrule`. This defers both the evaluation of the product rule and
    the projection until(/if) the tangent gets used.

