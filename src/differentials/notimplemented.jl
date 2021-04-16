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

extern(x::NotImplemented) = _error(x)

Base.iterate(x::NotImplemented) = _error(x)
Base.iterate(x::NotImplemented, ::Any) = _error(x)

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

Base.convert(::Type{<:Number}, x::NotImplemented) = _error(x)

_error(::NotImplemented) = error("differential not implemented")
function _error(x::NotImplemented{Module,LineNumberNode})
    error(
        "differential not implemented @ ",
        x.mod,
        " ",
        x.source,
        x.info === nothing ? "" : "\nInfo: " * x.info,
    )
end

"""
    @not_implemented(info=nothing)

Create a differential that indicates that the derivative is not implemented.

Optionally, you can provide additional information such as a Github issue
about the missing differential. Debugging information is only tracked and
displayed if `ChainRulesCore.debug_mode()` returns `true`.
"""
macro not_implemented(info=nothing)
    return if debug_mode()
        :(NotImplemented($__module__, $(QuoteNode(__source__)), $info))
    else
        :(NotImplemented())
    end
end
