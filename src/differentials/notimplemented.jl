"""
    NotImplemented

This differential indicates that the derivative is not implemented.
"""
struct NotImplemented{M,S,I} <: AbstractDifferential
    mod::M
    source::S
    info::I
end

extern(x::NotImplemented) = _error(x)

Base.zero(x::NotImplemented) = _error(x)
Base.zero(::Type{<:NotImplemented}) = _error(NotImplemented(nothing, nothing, nothing))

Base.iterate(x::NotImplemented) = _error(x)
Base.iterate(x::NotImplemented, ::Any) = _error(x)

Base.adjoint(x::NotImplemented) = _error(x)
Base.conj(x::NotImplemented) = _error(x)
Base.transpose(x::NotImplemented) = _error(x)

Base.:+(x::NotImplemented) = _error(x)
Base.:/(x::NotImplemented, ::Any) = _error(x)

Base.Broadcast.broadcastable(x::NotImplemented) = _error(x)

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
    mod = debug_mode() ? __module__ : nothing
    source = debug_mode() ? QuoteNode(__source__) : nothing
    return :(NotImplemented($mod, $source, $info))
end
