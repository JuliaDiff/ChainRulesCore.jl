module ChainRulesCoreSparseArraysExt

using ChainRulesCore
using ChainRulesCore: project_type, _projection_mismatch
using LinearAlgebra: Hermitian, Symmetric, tril, triu
using SparseArrays: SparseVector, SparseMatrixCSC, nzrange, rowvals, getcolptr, nonzeros

const HermSparse{T, I} = Hermitian{T, SparseMatrixCSC{T, I}}
const SymSparse{T, I} = Symmetric{T, SparseMatrixCSC{T, I}}
const HermOrSymSparse{T, I} = Union{HermSparse{T, I}, SymSparse{T, I}}

const SparseProjectToData{T, I} = NamedTuple{
    (:element, :axes, :rowval, :nzranges, :colptr),
    Tuple{
        ProjectTo{T, NamedTuple{(), Tuple{}}},
        Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},
        Vector{I},
        Vector{UnitRange{Int64}},
        Vector{I},
    },
}

const SparseProjectTo{T, I} = ProjectTo{SparseMatrixCSC, SparseProjectToData{T, I}}

const HermSparseProjectTo{T, I} = ProjectTo{
    Hermitian,
    NamedTuple{
        (:uplo, :parent),
        Tuple{Symbol, SparseProjectTo{T, I}},
    },
}

const SymSparseProjectTo{T, I} = ProjectTo{
    Symmetric,
    NamedTuple{
        (:uplo, :parent),
        Tuple{Symbol, SparseProjectTo{T, I}},
    },
}

ChainRulesCore.is_inplaceable_destination(::SparseVector) = true
ChainRulesCore.is_inplaceable_destination(::SparseMatrixCSC) = true

# Word from on high is that we should regard all un-stored values of sparse arrays as
# structural zeros. Thus ProjectTo needs to store nzind, and get only those.
# This implementation very naiive, can probably be made more efficient.

function ChainRulesCore.ProjectTo(x::SparseVector{T}) where {T<:Number}
    return ProjectTo{SparseVector}(;
        element=ProjectTo(zero(T)), nzind=x.nzind, axes=axes(x)
    )
end
function (project::ProjectTo{SparseVector})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        if size(dx, 1) != length(project.axes[1])
            throw(_projection_mismatch(project.axes, size(dx)))
        end
        reshape(dx, project.axes)
    end
    nzval = map(i -> project.element(dy[i]), project.nzind)
    return SparseVector(length(dx), project.nzind, nzval)
end
function (project::ProjectTo{SparseVector})(dx::SparseVector)
    if size(dx) != map(length, project.axes)
        throw(_projection_mismatch(project.axes, size(dx)))
    end
    # When sparsity pattern is unchanged, all the time is in checking this,
    # perhaps some simple hash/checksum might be good enough?
    samepattern = project.nzind == dx.nzind
    # samepattern = length(project.nzind) == length(dx.nzind)
    if eltype(dx) <: project_type(project.element) && samepattern
        return dx
    elseif samepattern
        nzval = map(project.element, dx.nzval)
        SparseVector(length(dx), dx.nzind, nzval)
    else
        nzind = project.nzind
        # Or should we intersect? Can this exploit sorting?
        # nzind = intersect(project.nzind, dx.nzind)
        nzval = map(i -> project.element(dx[i]), nzind)
        return SparseVector(length(dx), nzind, nzval)
    end
end

function ChainRulesCore.ProjectTo(x::SparseMatrixCSC{T}) where {T<:Number}
    return ProjectTo{SparseMatrixCSC}(;
        element=ProjectTo(zero(T)),
        axes=axes(x),
        rowval=rowvals(x),
        nzranges=nzrange.(Ref(x), axes(x, 2)),
        colptr=x.colptr,
    )
end
# You need not really store nzranges, you can get them from colptr -- TODO
# nzrange(S::AbstractSparseMatrixCSC, col::Integer) = getcolptr(S)[col]:(getcolptr(S)[col+1]-1)
function (project::ProjectTo{SparseMatrixCSC})(dx::AbstractArray)
    dy = if axes(dx) == project.axes
        dx
    else
        if size(dx) != (length(project.axes[1]), length(project.axes[2]))
            throw(_projection_mismatch(project.axes, size(dx)))
        end
        reshape(dx, project.axes)
    end
    nzval = Vector{project_type(project.element)}(undef, length(project.rowval))
    k = 0
    for col in project.axes[2]
        for i in project.nzranges[col]
            row = project.rowval[i]
            val = dy[row, col]
            nzval[k += 1] = project.element(val)
        end
    end
    m, n = map(length, project.axes)
    return SparseMatrixCSC(m, n, project.colptr, project.rowval, nzval)
end

function (project::ProjectTo{SparseMatrixCSC})(dx::SparseMatrixCSC)
    if size(dx) != map(length, project.axes)
        throw(_projection_mismatch(project.axes, size(dx)))
    end
    samepattern = dx.colptr == project.colptr && dx.rowval == project.rowval
    # samepattern = length(dx.colptr) == length(project.colptr) && dx.colptr[end] == project.colptr[end]
    if eltype(dx) <: project_type(project.element) && samepattern
        return dx
    elseif samepattern
        nzval = map(project.element, dx.nzval)
        m, n = size(dx)
        return SparseMatrixCSC(m, n, dx.colptr, dx.rowval, nzval)
    else
        invoke(project, Tuple{AbstractArray}, dx)
    end
end

#####
##### Hermitian/Symmetric sparse projection
#####

function project!(A::SparseMatrixCSC{T, I}, B::SparseMatrixCSC{<:Any, J}, uplo::Char) where {T, I, J}
    @assert size(A) == size(B)

    @inbounds for j in axes(A, 2)
        p = getcolptr(A)[j]
        pstop = getcolptr(A)[j + 1]
        q = getcolptr(B)[j]
        qstop = getcolptr(B)[j + 1]

        while p < pstop
            i = rowvals(A)[p]

            if (uplo == 'L' && i >= j) || (uplo == 'U' && i <= j)
                while q < qstop && rowvals(B)[q] < i
                    q += one(J)
                end

                if q < qstop && rowvals(B)[q] == i
                    nonzeros(A)[p] = nonzeros(B)[q]
                else
                    nonzeros(A)[p] = zero(T)
                end
            end

            p += one(I)
        end
    end

    return A
end

function project!(A::HermOrSymSparse, B::HermOrSymSparse)
    if A.uplo == B.uplo
        project!(parent(A), parent(B), A.uplo)
    elseif A.uplo == 'L'
        project!(parent(A), tril(B), A.uplo)
    else
        project!(parent(A), triu(B), A.uplo)
    end

    return A
end

function sparse_from_project(P::SparseProjectTo{T, I}) where {T, I}
    m, n = map(length, P.axes)
    return SparseMatrixCSC(m, n, P.colptr, P.rowval, zeros(T, length(P.rowval)))
end

function sparse_from_project(P::HermSparseProjectTo)
    return Hermitian(sparse_from_project(P.parent), P.uplo)
end

function sparse_from_project(P::SymSparseProjectTo)
    return Symmetric(sparse_from_project(P.parent), P.uplo)
end

function checkpatternsym(n, Acolptr::Vector{IA}, Bcolptr::Vector{IB}, Arowval::AbstractVector, Browval::AbstractVector, uplo::Char) where {IA, IB}
    for j in 1:n
        pa = Acolptr[j]
        pb = Bcolptr[j]
        pastop = Acolptr[j + 1]
        pbstop = Bcolptr[j + 1]

        while pa < pastop && pb < pbstop
            ia = Arowval[pa]
            ib = Browval[pb]

            if (uplo == 'L' && ia < j) || (uplo == 'U' && ia > j)
                pa += one(IA)
            elseif (uplo == 'L' && ib < j) || (uplo == 'U' && ib > j)
                pb += one(IB)
            elseif ia == ib
                pa += one(IA)
                pb += one(IB)
            else
                return false
            end
        end

        while pa < pastop
            ia = Arowval[pa]

            if (uplo == 'L' && ia >= j) || (uplo == 'U' && ia <= j)
                return false
            end

            pa += one(IA)
        end

        while pb < pbstop
            ib = Browval[pb]

            if (uplo == 'L' && ib >= j) || (uplo == 'U' && ib <= j)
                return false
            end

            pb += one(IB)
        end
    end

    return true
end

function checkpatternsym(P, dX)
    return false
end

function checkpatternsym(P::Union{HermSparseProjectTo{T, I}, SymSparseProjectTo{T, I}}, dX::HermOrSymSparse{T, I}) where {T, I}
    dXP = parent(dX)
    return Symbol(dX.uplo) == P.uplo && checkpatternsym(size(dXP, 2), P.parent.colptr, dXP.colptr, P.parent.rowval, dXP.rowval, dX.uplo)
end

function (P::HermSparseProjectTo{T, I})(dX::HermSparse) where {T, I}
    if checkpatternsym(P, dX)
        return dX
    else
        return project!(sparse_from_project(P), dX)
    end
end

function (P::SymSparseProjectTo{T, I})(dX::SymSparse) where {T, I}
    if checkpatternsym(P, dX)
        return dX
    else
        return project!(sparse_from_project(P), dX)
    end
end

function (P::HermSparseProjectTo{T, I})(dX::SymSparse{T, I}) where {T <: Real, I}
    if checkpatternsym(P, dX)
        return Hermitian(parent(dX), P.uplo)
    else
        return project!(sparse_from_project(P), dX)
    end
end

function (P::SymSparseProjectTo{T, I})(dX::HermSparse{T, I}) where {T <: Real, I}
    if checkpatternsym(P, dX)
        return Symmetric(parent(dX), P.uplo)
    else
        return project!(sparse_from_project(P), dX)
    end
end

end # module
