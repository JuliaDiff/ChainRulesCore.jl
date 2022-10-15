"""
    @not_implemented(info)

Create a tangent that indicates that the derivative is not implemented.

The `info` should be useful information about the missing tangent for debugging.

!!! note
    This macro should be used only if the automatic differentiation would error
    otherwise. It is mostly useful if the function has multiple inputs or outputs,
    and one has worked out analytically and implemented some but not all tangents.

!!! note
    It is good practice to include a link to a GitHub issue about the missing
    tangent in the debugging information.
"""
macro not_implemented(info)
    return :(NotImplemented($__module__, $(QuoteNode(__source__)), $(esc(info))))
end

"""
    NotImplemented

This tangent indicates that the derivative is not implemented.

It is generally best to construct this using the [`@not_implemented`](@ref) macro,
which will automatically insert the source module and file location.
"""
struct NotImplemented <: AbstractTangent
    mod::Module
    source::LineNumberNode
    info::String
end

# required for `@scalar_rule`
# (together with `conj(x::AbstractTangent) = x` and the definitions in
# tangent_arithmetic.jl)
Base.Broadcast.broadcastable(x::NotImplemented) = Ref(x)

# throw error with debugging information for other standard information
# (`+`, `-`, `*`, and `dot` are defined in tangent_arithmetic.jl)
Base.:/(x::NotImplemented, ::Any) = throw(NotImplementedException(x))
Base.:/(::Any, x::NotImplemented) = throw(NotImplementedException(x))
Base.:/(x::NotImplemented, ::NotImplemented) = throw(NotImplementedException(x))

# Fix method ambiguity errors (#589)
Base.:/(x::AbstractZero, ::NotImplemented) = x
Base.:/(x::NotImplemented, ::AbstractThunk) = throw(NotImplementedException(x))
Base.:/(::AbstractThunk, x::NotImplemented) = throw(NotImplementedException(x))

Base.zero(x::NotImplemented) = throw(NotImplementedException(x))
function Base.zero(::Type{<:NotImplemented})
    return throw(
        NotImplementedException(
            @not_implemented(
                "`zero` is not defined for missing tangents of type `NotImplemented`"
            )
        ),
    )
end

Base.iterate(x::NotImplemented) = throw(NotImplementedException(x))
Base.iterate(x::NotImplemented, ::Any) = throw(NotImplementedException(x))

Base.adjoint(x::NotImplemented) = throw(NotImplementedException(x))
Base.transpose(x::NotImplemented) = throw(NotImplementedException(x))

Base.convert(::Type{<:Number}, x::NotImplemented) = throw(NotImplementedException(x))

function Base.show(io::IO, x::NotImplemented)
    return print(io, "NotImplemented(", x.mod, ", ", x.source, ", ", x.info, ")")
end

struct NotImplementedException <: Exception
    mod::Module
    source::LineNumberNode
    info::String
end

function NotImplementedException(x::NotImplemented)
    return NotImplementedException(x.mod, x.source, x.info)
end

function Base.showerror(io::IO, e::NotImplementedException)
    print(io, "tangent not implemented @ ", e.mod, " ", e.source)
    if e.info !== nothing
        print(io, "\nInfo: ", e.info)
    end
    return nothing
end
