"""
    NotImplemented

This differential indicates that the derivative is not implemented.
"""
struct NotImplemented{M,S,I} <: AbstractDifferential
    mod::M
    source::S
    info::I
end

NotImplemented() = NotImplemented(nothing, nothing, nothing)

extern(x::NotImplemented) = throw(NotImplementedException(x))

Base.iterate(x::NotImplemented) = throw(NotImplementedException(x))
Base.iterate(x::NotImplemented, ::Any) = throw(NotImplementedException(x))

Base.:+(x::NotImplemented, ::Any) = x
Base.:+(::Any, x::NotImplemented) = x
Base.:+(x::NotImplemented, ::NotImplemented) = x
Base.:*(x::NotImplemented, ::Any) = x
Base.:*(::Any, x::NotImplemented) = x
Base.:*(x::NotImplemented, ::NotImplemented) = x

# Linear operators
Base.adjoint(x::NotImplemented) = x
Base.transpose(x::NotImplemented) = x

Base.Broadcast.broadcastable(x::NotImplemented) = Ref(x)

Base.convert(::Type{<:Number}, x::NotImplemented) = throw(NotImplementedException(x))

function Base.show(io::IO, ::NotImplemented{Nothing,Nothing,Nothing})
    return print(io, "NotImplemented()")
end
function Base.show(io::IO, x::NotImplemented)
    return print(io, "NotImplemented(", x.mod, ", ", x.source, ", ", x.info, ")")
end

"""
    @not_implemented(info=nothing)

Create a differential that indicates that the derivative is not implemented.

Optionally, one can provide additional information about the missing differential.
Debugging information is only tracked and displayed if `ChainRulesCore.debug_mode()`
returns `true`.

!!! note
    This macro should be used only if the automatic differentiation would error
    otherwise. It is mostly useful if the function has multiple inputs and one
    has worked out analytically differentials of some but not all of them.

!!! note
    It is good practice to provide a link to a GitHub issue about the missing
    differential as additional debugging information.
"""
macro not_implemented(info=nothing)
    return if debug_mode()
        :(NotImplemented($__module__, $(QuoteNode(__source__)), $info))
    else
        :(NotImplemented())
    end
end

struct NotImplementedException{T<:NotImplemented} <: Exception
    x::T
end

function Base.showerror(io::IO, e::NotImplementedException)
    return print(io, "differential not implemented")
end
function Base.showerror(
    io::IO,
    e::NotImplementedException{<:NotImplemented{Module,LineNumberNode}},
)
    x = e.x
    return print(
        io,
        "differential not implemented @ ",
        x.mod,
        " ",
        x.source,
        x.info === nothing ? "" : "\nInfo: " * x.info,
    )
end
