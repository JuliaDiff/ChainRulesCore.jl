# Fallbacks for destructure
destructure(X::AbstractArray) = collect(X)

pushforward_of_destructure(X) = dX -> frule((NoTangent(), dX), destructure, X)[2]

function pullback_of_destructure(config::RuleConfig, X)
    return dY -> rrule_via_ad(config, destructure, X)[2](dY)[2]
end

# Restructure machinery.
struct Restructure{P, D}
    data::D
end

function pullback_of_restructure(config::RuleConfig, X)
    return dY -> rrule_via_ad(config, Restructure(X), destructure(X))[2](dY)[2]
end




# Array

function pullback_of_destructure(X::Array{<:Real})
    pullback_destructure_Array(X̄::AbstractArray{<:Real}) = X̄
    return pullback_destructure_Array
end


function pullback_of_restructure(X::Array{<:Real})
    pullback_restructure_Array(X̄::AbstractArray{<:Real}) = X̄
    return pullback_restructure_Array
end

function pullback_of_destructure(config::RuleConfig, X::Array{<:Real})
    return pullback_of_destructure(X)
end

function pullback_of_restructure(config::RuleConfig, X::Array{<:Real})
    return pullback_of_destructure(X)
end


# Stuff below here for Array to move to tests.
destructure(X::Array) = X

frule((_, dX)::Tuple{Any, AbstractArray}, ::typeof(destructure), X::Array) = X, dX

function rrule(::typeof(destructure), X::Array)
    destructure_pullback(dXm::AbstractArray) = NoTangent(), dXm
    return X, destructure_pullback
end

Restructure(X::P) where {P<:Array} = Restructure{P, Nothing}(nothing)

(r::Restructure{P})(X::Array) where {P<:Array} = X

function frule(
    (_, dX)::Tuple{Any, AbstractArray}, ::Restructure{P}, X::Array,
) where {P<:Array}
    return X, dX
end

function rrule(::Restructure{P}, X::Array) where {P<:Array}
    restructure_pullback(dY::AbstractArray) = NoTangent(), dY
    return X, restructure_pullback
end





# Diagonal
function pullback_of_destructure(D::P) where {P<:Diagonal}
    pullback_destructure_Diagonal(D̄::AbstractArray) = Tangent{P}(diag=diag(D̄))
    return pullback_destructure_Diagonal
end

function pullback_of_restructure(D::P) where {P<:Diagonal}
    pullback_restructure_Diagonal(D̄::Tangent) = Diagonal(D̄.diag)
    return pullback_restructure_Diagonal
end

function pullback_of_destructure(config::RuleConfig, X::Diagonal)
    return pullback_of_destructure(X)
end

function pullback_of_restructure(config::RuleConfig, X::Diagonal)
    return pullback_of_destructure(X)
end

# Stuff below here for Diagonal to move to tests.
destructure(X::Diagonal) = collect(X)

function frule((_, dX)::Tuple{Any, Tangent}, ::typeof(destructure), X::Diagonal)
    des_diag, d_des_diag = frule((NoTangent(), dX.diag), destructure, X.diag)
    return collect(X), collect(Diagonal(d_des_diag))
end

function rrule(::typeof(destructure), X::P) where {P<:Diagonal}
    _, des_diag_pb = rrule(destructure, X.diag)
    function destructure_pullback(dY::AbstractMatrix)
        d_des_diag = diag(dY)
        _, d_diag = des_diag_pb(d_des_diag)
        return NoTangent(), Tangent{P}(diag=d_diag)
    end
    return destructure(X), destructure_pullback
end

Restructure(X::P) where {P<:Diagonal} = Restructure{P, Nothing}(nothing)

function (r::Restructure{P})(X::Array) where {P<:Diagonal}
    @assert isdiag(X) # for illustration. Remove in actual because numerics.
    return Diagonal(diag(X))
end

function frule(
    (_, dX)::Tuple{Any, AbstractArray}, ::Restructure{P}, X::Array,
) where {P<:Diagonal}
    return Diagonal(diag(X)), Tangent{P}(diag=diag(dX))
end

function rrule(::Restructure{P}, X::Array) where {P<:Diagonal}
    restructure_pullback(dY::Tangent) = NoTangent(), Diagonal(dY.diag)
    return X, restructure_pullback
end





# Symmetric
function pullback_of_destructure(S::P) where {P<:Symmetric}
    function destructure_pullback_Symmetric(dXm::AbstractMatrix)
        U = UpperTriangular(dXm)
        L = LowerTriangular(dXm)
        if S.uplo == 'U'
            return Tangent{P}(data=U + L' - Diagonal(dXm))
        else
            return Tangent{P}(data=U' + L - Diagonal(dXm))
        end
    end
    return destructure_pullback_Symmetric
end

# Assume upper-triangular for now.
function pullback_of_restructure(S::P) where {P<:Symmetric}
    function restructure_pullback_Symmetric(dY::Tangent)
        return collect(UpperTriangular(dY.data))
    end
    return restructure_pullback_Symmetric
end

function pullback_of_destructure(config::RuleConfig, X::Symmetric)
    return pullback_of_destructure(X)
end

function pullback_of_restructure(config::RuleConfig, X::Symmetric)
    return pullback_of_destructure(X)
end

# Stuff below here for Symmetric to move to tests.
function destructure(X::Symmetric)
    des_data = destructure(X.data)
    if X.uplo == 'U'
        U = UpperTriangular(des_data)
        return U + U' - Diagonal(des_data)
    else
        L = LowerTriangular(des_data)
        return L' + L - Diagonal(X.data)
    end
end

# This gives you the natural tangent!
function frule((_, dx)::Tuple{Any, Tangent}, ::typeof(destructure), x::Symmetric)
    des_data, d_des_data = frule((NoTangent(), dx.data), destructure, x.data)

    if x.uplo == 'U'
        dU = UpperTriangular(d_des_data)
        return destructure(x), dU + dU' - Diagonal(d_des_data)
    else
        dL = LowerTriangular(d_des_data)
        return destructure(x), dL + dL' - Diagonal(d_des_data)
    end
end

function rrule(::typeof(destructure), X::P) where {P<:Symmetric}
    function destructure_pullback(dXm::AbstractMatrix)
        U = UpperTriangular(dXm)
        L = LowerTriangular(dXm)
        if X.uplo == 'U'
            return NoTangent(), Tangent{P}(data=U + L' - Diagonal(dXm))
        else
            return NoTangent(), Tangent{P}(data=U' + L - Diagonal(dXm))
        end
    end
    return destructure(X), destructure_pullback
end

Restructure(X::P) where {P<:Symmetric} = Restructure{P, P}(X)

# In generic-abstractarray-rrule land, assume getindex was used, so the
# strict-lower-triangle was never touched.
function (r::Restructure{P})(X::Array) where {P<:Symmetric}
    strict_lower_triangle_of_data = LowerTriangular(r.data.data) - Diagonal(r.data.data)
    return Symmetric(UpperTriangular(X) + strict_lower_triangle_of_data)
end

function frule(
    (_, dX)::Tuple{Any, AbstractArray}, r::Restructure{P}, X::Array,
) where {P<:Symmetric}
    return r(X), Tangent{P}(data=UpperTriangular(dX))
end

function rrule(r::Restructure{P}, X::Array) where {P<:Symmetric}
    function restructure_pullback(dY::Tangent)
        d_restructure = Tangent{Restructure{P}}(data=Tangent{P}(data=tril(dY.data)))
        return d_restructure, collect(UpperTriangular(dY.data))
    end
    return r(X), restructure_pullback
end





# Cholesky -- you get to choose whatever destructuring operation is helpful for a given
# type. This one is helpful for writing generic pullbacks for `cholesky`, the output of
# which is a Cholesky. Not completed. Probably won't be included in initial merge.
destructure(C::Cholesky) = Cholesky(destructure(C.factors), C.uplo, C.info)

# Restructure(C::P) where {P<:Cholesky} = Restructure{P, Nothing}()
