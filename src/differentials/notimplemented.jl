"""
    NotImplemented

This differential indicates that the derivative is not implemented.
"""
struct NotImplemented{M,S,I} <: AbstractDifferential
    mod::M
    source::S
    info::I
end

# required for `@scalar_rule`
# (together with `conj(x::AbstractDifferential) = x` and the definitions in
# differential_arithmetic.jl)
Base.Broadcast.broadcastable(x::NotImplemented) = Ref(x)

# throw error with debugging information for other standard information
# (`+`, `-`, `*`, and `dot` are defined in differential_arithmetic.jl)
extern(x::NotImplemented) = throw(NotImplementedException(x))

Base.:/(x::NotImplemented, ::Any) = throw(NotImplementedException(x))
Base.:/(::Any, x::NotImplemented) = throw(NotImplementedException(x))
Base.:/(x::NotImplemented, ::NotImplemented) = throw(NotImplementedException(x))

Base.zero(x::NotImplemented) = throw(NotImplementedException(x))
Base.zero(::Type{<:NotImplemented}) = throw(NotImplementedException())

Base.iterate(x::NotImplemented) = throw(NotImplementedException(x))
Base.iterate(x::NotImplemented, ::Any) = throw(NotImplementedException(x))

Base.adjoint(x::NotImplemented) = throw(NotImplementedException(x))
Base.transpose(x::NotImplemented) = throw(NotImplementedException(x))

Base.convert(::Type{<:Number}, x::NotImplemented) = throw(NotImplementedException(x))

function Base.show(io::IO, x::NotImplemented)
    return print(io, "NotImplemented(", x.mod, ", ", x.source, ", ", x.info, ")")
end

"""
    @not_implemented(info=nothing)

Create a differential that indicates that the derivative is not implemented.

Optionally, one can provide additional information about the missing differential.

!!! note
    This macro should be used only if the automatic differentiation would error
    otherwise. It is mostly useful if the function has multiple inputs and one
    has worked out analytically differentials of some but not all of them.

!!! note
    It is good practice to provide a link to a GitHub issue about the missing
    differential as additional debugging information.
"""
macro not_implemented(info=nothing)
    return :(NotImplemented($__module__, $(QuoteNode(__source__)), $info))
end

struct NotImplementedException{M,S,I} <: Exception
    mod::M
    source::S
    info::I
end

NotImplementedException() = NotImplementedException(nothing, nothing, nothing)
function NotImplementedException(x::NotImplemented)
    return NotImplementedException(x.mod, x.source, x.info)
end

function Base.showerror(io::IO, e::NotImplementedException)
    return print(io, "differential not implemented")
end
function Base.showerror(io::IO, e::NotImplementedException{Module,LineNumberNode})
    print(io, "differential not implemented @ ", e.mod, " ", e.source)
    if e.info !== nothing
        print(io, "\nInfo: ", e.info)
    end
    return
end
