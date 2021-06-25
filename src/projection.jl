using LinearAlgebra: Diagonal, diag


"""
    preproject(x)

Returns a NamedTuple containing information needed to [`project`](@ref) a differential `dx`
onto the type `T` for a primal `x`. For example, when projecting `dx=ZeroTangent()` on an
array `T=Array{T, N}`, the size of `x` is not available from `T`.
"""
function preproject end

preproject(x::Array) = (; size=size(x), eltype=eltype(x))

preproject(x::Diagonal{<:Any, V}) where {V} = (; Vinfo=preproject(diag(x)))

preproject(x::Symmetric{<:Any, M}) where {M} = (; uplo=Symbol(x.uplo), Minfo=preproject(parent(x)))

"""
    project(T::Type, dx; [info])

Projects the differential `dx` for primal `x` onto type `T`. The kwarg `info` contains
information about the primal `x` that is needed to project onto `T`, e.g. the size of an
array. It is obtained from `preproject(x)`.
"""
function project end

# fallback (structs)
project(::Type{T}, dx::T) where T = dx
project(::Type{T}, dx::AbstractZero) where T = zero(T)
project(::Type{T}, dx::AbstractThunk) where T = project(T, unthunk(dx))
function project(::Type{T}, dx::Tangent{<:T}) where T
    fnames = fieldnames(T)
    values = [getproperty(dx, fn) for fn in fnames]
    return T((; zip(fnames, values)...)...)
end # TODO: make Tangent work recursively

# Real
project(::Type{T}, dx::Real) where {T<:Real} = T(dx)
project(::Type{T}, dx::Number) where {T<:Real} = T(real(dx))
project(::Type{T}, dx::AbstractZero) where {T<:Real} = zero(T)
project(::Type{T}, dx::AbstractThunk) where {T<:Real} = project(T, unthunk(dx))
# Number
project(::Type{T}, dx::Number) where {T<:Number} = T(dx)
project(::Type{T}, dx::AbstractZero) where {T<:Number} = zero(T)
project(::Type{T}, dx::AbstractThunk) where {T<:Number} = project(T, unthunk(dx))

# Arrays
project(AT::Type{Array{T, N}}, dx::Array{T, N}; info) where {T, N} = dx
project(AT::Type{Array{T, N}}, dx::AbstractArray; info) where {T, N} = project(AT, collect(dx); info=info)
project(AT::Type{Array{T, N}}, dx::Array; info) where {T, N} = project.(T, dx)
project(AT::Type{Array{T, N}}, dx::AbstractZero; info) where {T, N} = zeros(T, info.size...)
project(AT::Type{Array{T, N}}, dx::AbstractThunk; info) where {T, N} = project(AT, unthunk(dx); info=info)

# Diagonal
project(DT::Type{<:Diagonal{<:Any, V}}, dx::AbstractMatrix; info) where {V} = Diagonal(project(V, diag(dx); info=info.Vinfo))
project(DT::Type{<:Diagonal{<:Any, V}}, dx::Tangent; info) where {V} = Diagonal(project(V, dx.diag; info=info.Vinfo))
project(DT::Type{<:Diagonal{<:Any, V}}, dx::AbstractZero; info) where {V} = Diagonal(project(V, dx; info=info.Vinfo))
project(DT::Type{<:Diagonal{<:Any, V}}, dx::AbstractThunk; info) where {V} = project(DT, unthunk(dx); info=info)

# Symmetric
project(ST::Type{<:Symmetric{<:Any, M}}, dx::AbstractMatrix; info) where {M} = Symmetric(project(M, dx; info=info.Minfo), info.uplo)
project(ST::Type{<:Symmetric{<:Any, M}}, dx::Tangent; info) where {M} = Symmetric(project(M, dx.data; info=info.Minfo), info.uplo)
project(ST::Type{<:Symmetric{<:Any, M}}, dx::AbstractZero; info) where {M} = Symmetric(project(M, dx; info=info.Minfo), info.uplo)
project(ST::Type{<:Symmetric{<:Any, M}}, dx::AbstractThunk; info) where {M} = project(ST, unthunk(dx); info=info)
