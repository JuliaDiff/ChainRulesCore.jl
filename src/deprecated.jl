Base.@deprecate_binding NO_FIELDS NoTangent()

const EXTERN_DEPRECATION = "`extern` is deprecated, use `unthunk` or `backing` instead, " *
    "depending on the use case."

"""
    extern(x)

Makes a best effort attempt to convert a differential into a primal value.
This is not always a well-defined operation.
For two reasons:
 - It may not be possible to determine the primal type for a given differential.
 For example, `Zero` is a valid differential for any primal.
 - The primal type might not be a vector space, thus might not be a valid differential type.
 For example, if the primal type is `DateTime`, it's not a valid differential type as two
 `DateTime` can not be added (fun fact: `Milisecond` is a differential for `DateTime`).

Where it is defined the operation of `extern` for a primal type `P` should be
`extern(x) = zero(P) + x`.

!!! note
    Because of its limitations, `extern` should only really be used for testing.
    It can be useful, if you know what you are getting out, as it recursively removes
    thunks, and otherwise makes outputs more consistent with finite differencing.

    The more useful action in general is to call `+`, or in the case of a [`Thunk`](@ref)
    to call [`unthunk`](@ref).

!!! warning
    `extern` may return an alias (not necessarily a copy) to data
    wrapped by `x`, such that mutating `extern(x)` might mutate `x` itself.
"""
@inline function extern(x)
    Base.depwarn(EXTERN_DEPRECATION, :extern)
    return x
end

extern(x::ZeroTangent) = (Base.depwarn(EXTERN_DEPRECATION, :extern); return false)  # false is a strong 0. E.g. `false * NaN = 0.0`

function extern(x::NoTangent)
    Base.depwarn(EXTERN_DEPRECATION, :extern)
    throw(ArgumentError("Derivative does not exit. Cannot be converted to an external type."))
end

extern(comp::Tangent) = (Base.depwarn(EXTERN_DEPRECATION, :extern); return backing(map(extern, comp)))  # gives a NamedTuple or Tuple

extern(x::NotImplemented) = (Base.depwarn(EXTERN_DEPRECATION, :extern); throw(NotImplementedException(x)))

@inline extern(x::AbstractThunk) = (Base.depwarn(EXTERN_DEPRECATION, :extern); return extern(unthunk(x)))

for T in (:Thunk, :InplaceableThunk)
    @eval function (x::$T)()
        Base.depwarn("`(x::" * string($T) * ")()` is deprecated, use `unthunk(x)`", Symbol(:call_, $(T))) 
        return unthunk(x)
    end    
end
