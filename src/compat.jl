if VERSION < v"1.2"
    Base.getproperty(x::Tuple, f::Int) = getfield(x, f)
end

if VERSION < v"1.1"
    # Note: these are actually *better* than the ones in julia 1.1, 1.2, 1.3,and 1.4
    # See: https://github.com/JuliaLang/julia/issues/34292
    function fieldtypes(::Type{T}) where {T}
        if @generated
            ntuple(i -> fieldtype(T, i), fieldcount(T))
        else
            ntuple(i -> fieldtype(T, i), fieldcount(T))
        end
    end

    function fieldnames(::Type{T}) where {T}
        if @generated
            ntuple(i -> fieldname(T, i), fieldcount(T))
        else
            ntuple(i -> fieldname(T, i), fieldcount(T))
        end
    end
end
