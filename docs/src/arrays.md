# Deriving Array Rules

When the inputs and outputs of the functions are arrays (potentially with scalars), a modified version of Giles' method is a quick way of deriving [`frule`](@ref)s and [`rrule`](@ref)s.
Giles' method is succinctly explained in [this paper](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf), but we will generalize it to handle arrays with both real and complex entries and arrays of arbitrary dimensions.

Throughout this tutorial, we will ust the following type alias

```julia
const RealOrComplex = Union{Real,Complex}
```

## Deriving forward-mode rules

Given a function

```julia
C = f(A::Array{<:RealOrComplex}, B::Array{<:RealOrComplex})::Array{<:RealOrComplex}
```

we write the differential of the output in terms of differentials of the inputs as

```math
\begin{equation}\label{cdiff}
dC = \frac{\partial f}{\partial A} dA + \frac{\partial f}{\partial B} dB.
\end{equation}
```

Notationally, we write the pushforward by replacing the differentials with the corresponding forward-mode sensitivities (e.g. replace $dC$ with $\dot{C}$):

$$\dot{C} = \frac{\partial f}{\partial A} \dot{A} + \frac{\partial f}{\partial B} \dot{B}.$$

The terms $\frac{\partial f}{\partial A}$ are array-array derivatives (i.e. a type of Jacobian).
We do not write these down explicitly, but we instead use differential identities to derive the terms $\frac{\partial f}{\partial A} dA$, which as we've seen behave like the Jacobian-vector-products $\frac{\partial f}{\partial A} \dot{A}$.
The differential identities follow directly from the usual scalar identities.
We will look at a few examples.

### Matrix addition

```julia
C = A + B
```

This one is easy:

$$C = A + B$$

$$dC = dA + dB$$

We can implement the `frule` in ChainRules` notation:

```julia
function frule(
    (_, ΔA, ΔB),
    ::typeof(+), A::Array{<:RealOrComplex}, B::Array{<:RealOrComplex},
)
    return (A + B, ΔA + ΔB)
end
```

### Matrix multiplication

```julia
C = A * B
```

$$C = A B$$

First we write in component form:

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

Then we use the product rule to get the scalar differential identity:

```math
\begin{align*}
dC_{ij} &= \sum_k \left( dA_{ik}~ B_{kj} + A_{ik} ~dB_{kj} \right)
            && \text{apply scalar product rule } d(x y) = dx~ y + x ~dy \\
        &= \sum_k dA_{ik}~ B_{kj} + \sum_k A_{ik} ~dB_{kj}
            && \text{expand sum}
\end{align*}
```

But the last expression is just a sum of matrix products:

```math
\begin{equation}\label{diffprod}
dC = dA~ B + A ~dB
\end{equation}
```

So we now have the matrix product rule, whose `frule` is

```julia
function frule(
    (_, ΔA, ΔB),
    ::typeof(*), A::Matrix{<:RealOrComplex}, B::Matrix{<:RealOrComplex},
)
    return (A * B, ΔA * B + A * ΔB)
end
```

### Matrix inversion

```julia
C = inv(A)
```

$$C = A^{-1}$$

It's easiest to derive this rule by writing either constraint:

```math
\begin{align*}
C A &= A^{-1} A = I\\
A C &= A A^{-1} = I,
\end{align*}
```

where $I$ is the identity matrix.

Now we use the matrix product rule to differentiate the constraint:

$$dC~ A + C ~dA = 0$$

We right-multiply both sides by $A^{-1}$ to isolate $dC$:

```math
\begin{align}
0  &= dC~ A A^{-1} + C ~dA~ A^{-1} && \nonumber\\
   &= dC~ I + C ~dA~ A^{-1} && \text{apply } A A^{-1} = I \nonumber\\
   &= dC + C ~dA~ C && \text{substitute } A^{-1} = C \nonumber\\
dC &= -C ~dA~ C && \text{solve for } dC \label{invdiff}
\end{align}
```

We write the `frule` as

```julia
function frule((_, ΔA), ::typeof(inv), A::Matrix{<:RealOrComplex})
    C = inv(A)
    ∂C = -C * ΔA * C
    return (C, ∂C)
end
```

### Other useful identities

Two useful identities are $d(A^H) = dA^H$ and $d(A^T) = dA^T$, where $\cdot^H$ is the conjugate transpose (i.e. the `adjoint` function).

## Deriving reverse-mode rules

Reverse-mode rules are a little more involved.
For a real scalar function $s = g(C)$, the differential of $s$ is the real part of the Frobenius inner product (`LinearAlgebra.dot`) of the adjoint of $C$, $\overline{C}$, and the differential of $C$:

```math
ds = \operatorname{real}\left( \langle \overline{C}, dC \rangle \right)
   = \operatorname{real}\left( \sum_{i,\dots,j} \operatorname{conj}(\overline{C}_{i,\dots,j}) ~dC_{i,\dots,j} \right),
```

where $\operatorname{conj}(\cdot)$ is the complex conjugate (`conj`), and $\operatorname{real}(\cdot)$ is the real part of its argument (`real`).

For matrices and vectors, we can write this as

$$ds = \operatorname{real}(\operatorname{tr}(\overline{C}^H dC)),$$

where $\operatorname{tr}$ is the matrix trace (`LinearAlgebra.tr`) function.

We can write the corresponding expression for $dA$ and $dB$ as

```math
\begin{align*}
ds &= \operatorname{real}(\operatorname{tr}(\overline{C}^H dC)) &&\\
   &= \operatorname{real}\left( \operatorname{tr}\left(
          \overline{C}^H \frac{\partial f}{\partial A} dA +
          \overline{C}^H \frac{\partial f}{\partial B} dB
      \right) \right) && \text{substitute } dC \text{ from } \eqref{cdiff}\\
   &= \operatorname{real}\left( \operatorname{tr}\left(
          \overline{C}^H \frac{\partial f}{\partial A} dA
      \right) \right) +
      \operatorname{real}\left( \operatorname{tr}\left(
          \overline{C}^H \frac{\partial f}{\partial B} dB
      \right) \right) && \text{expand using } \eqref{trexpand}
\end{align*}
```

By applying the same definition of $ds$ to the intermediates $A$ and $B$, we get

```math
ds = \operatorname{real}\left( \operatorname{tr}(\overline{A}^H  dA) \right) +
     \operatorname{real}\left( \operatorname{tr}(\overline{B}^H ~dB) \right)
```

Combining these two identities and solving for $\overline{A}$ and $\overline{B}$ gives us

```math
\begin{align*}
    \overline{A} &= (\overline{A}^H)^H
                  = \left( \overline{C}^H \frac{\partial f}{\partial A} \right)^H
                  = \left( \frac{\partial f}{\partial A} \right)^H \overline{C}\\
    \overline{B} &= \left( \frac{\partial f}{\partial B} \right)^H \overline{C}
\end{align*}
```

Giles' method for deriving pullback functions is to first derive the differential identity (i.e. pushforward) using the above approach, then pre-multiply by $\overline{C}^H$, and take the real trace.
Subsequently, manipulate into this form and solve for the adjoint derivatives of the inputs.
Several properties of the trace function make this easier:

```math
\begin{align}
    \operatorname{tr}(A+B) &= \operatorname{tr}(A) + \operatorname{tr}(B) \label{trexpand}\\
    \operatorname{tr}(A^T) &= \operatorname{tr}(A) \nonumber\\
    \operatorname{tr}(A^H) &= \operatorname{conj}(\operatorname{tr}(A)) \nonumber\\
    \operatorname{tr}(AB) &= \operatorname{tr}(BA) \label{trperm}
\end{align}
```

!!! note
    Our method is identical to Giles' approach, except we have replace the transpose with the conjugate transpose, and we have added the constraint that the inner product be real.
    This produces the correct pullbacks for arrays with complex entries.

Here are a few examples.

### Matrix multiplication

```julia
C = A * B
```

We above derived in \eqref{diffprod} the differential identity

$$dC = dA~ B + A ~dB$$

We now multiply by $\overline{C}^H$ and take the real trace:

```math
\begin{align*}
ds &= \operatorname{real}\left( \operatorname{tr}(\overline{C}^H ~dC) \right) &&\\
   &= \operatorname{real}\left( \operatorname{tr}(\overline{C}^H ~\left(
          dA~ B + A ~dB
      \right)) \right) &&
        \text{substitute } dC \text{ from } \eqref{diffprod}\\
   &= \operatorname{real}\left( \operatorname{tr}(\overline{C}^H ~dA~ B) \right) +
      \operatorname{real}\left( \operatorname{tr}(\overline{C}^H  A ~dB) \right) &&
        \text{expand using } \eqref{trexpand} \\
   &= \operatorname{real}\left( \operatorname{tr}(B \overline{C}^H ~dA) \right) +
      \operatorname{real}\left( \operatorname{tr}(\overline{C}^H A ~dB) \right) &&
        \text{rearrange the left side using } \eqref{trperm}\\
   &= \operatorname{real}\left( \operatorname{tr}(\overline{A}^H  dA) \right) +
      \operatorname{real}\left( \operatorname{tr}(\overline{B}^H ~dB) \right) && \\
\end{align*}
```

That's it!
The expression is in the desired form to solve for the adjoints:

```math
\begin{align*}
    \overline{A} &= \left( \overline{A}^H \right)^H = \left( B \overline{C}^H \right)^H = \overline{C} B^H\\
    \overline{B} &= \left( \overline{B}^H \right)^H = \left( \overline{C}^H A \right)^H = A^H \overline{C}
\end{align*}
```

Using ChainRules' notation, we would implement the `rrule` as

```julia
function rrule(::typeof(*), A::Matrix{<:RealOrComplex}, B::Matrix{<:RealOrComplex})
    function times_pullback(ΔC)
        ∂A = @thunk(ΔC * B')
        ∂B = @thunk(A' * ΔC)
        return (NO_FIELDS, ∂A, ∂B)
    end
    return A * B, times_pullback
end
```

### Matrix inversion

```julia
C = inv(A)
```

In \eqref{invdiff}, we derived the differential identity as

$$dC = -C ~dA~ C$$

Multiplying by $\overline{C}^H$ and taking the real trace,

```math
    ds = \operatorname{real}\left( \operatorname{tr}(\overline{C}^H ~dC) \right)
       = \operatorname{real}\left( \operatorname{tr}(-\overline{C}^H C ~dA~ C) \right)
```

Applying the trace identity \eqref{trperm} to manipulate into the desired form,

$$ds = \operatorname{real}(\operatorname{tr}(-C \overline{C}^H C ~dA) ),$$

we can now solve for $\overline{A}$:

$$\overline{A} = (-C \overline{C}^H C)^H = -C^H \overline{C} C^H$$

We can implement the resulting `rrule` as:

```julia
function rrule(::typeof(inv), A::Matrix{<:RealOrComplex})
    C = inv(A)
    function inv_pullback(ΔC)
        ∂A = -C' * ΔC * C'
        return (NO_FIELDS, ∂A)
    end
    return C, inv_pullback
end
```

## More examples

For more instructive examples of matrix and vector rules, see [Giles' paper](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf) and the [LinearAlgebra rules in ChainRules](https://github.com/JuliaDiff/ChainRules.jl/tree/master/src/rulesets/LinearAlgebra).

## Generalizing to multidimensional arrays

For both forward- and reverse-mode rules, the first step was to write down the differential identities, which followed directly from the scalar differential identities.
This approach follows for arrays, but it's easier to work in component form.
Consider the following function

```julia
C = sum(abs2, A::Array{<:RealOrComplex,3}; dims=2)::Array{<:Real,3}
```

which we write as

$$C_{i1k} = \sum_{j} |A_{ijk}|^2 = \sum_{j} \operatorname{real}(\operatorname{conj}(A_{ijk}) A_{ijk})$$

The differential identity is

```math
\begin{align*}
    dC_{i1k} &= \sum_j \operatorname{real}( \operatorname{conj}(dA_{ijk})~ A_{ijk} +
                            \operatorname{conj}(A_{ijk}) ~dA_{ijk} ) \\
             &= \sum_j \operatorname{real}( \operatorname{conj}\left( \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right) +
                                                      \operatorname{conj}(A_{ijk}) ~dA_{ijk} )\\
             &= \sum_j 2 \operatorname{real}( \operatorname{conj}(A_{ijk}) ~dA_{ijk} )
\end{align*}
```

where in the last step we have used the fact that for all real $a$ and $b$,

$$(a + i b) + \operatorname{conj}(a + i b) = (a + i b) + (a - i b) = 2 a = 2 \operatorname{real} (a + i b).$$

The `frule` can be implemented generally as

```julia
function frule(
    (_, _, ΔA),
    ::typeof(sum), ::typeof(abs2), A::Array{<:RealOrComplex};
    dims = :,
)
    C = sum(abs2, A; dims = dims)
    ∂C = sum(2 .* real.(conj.(A) .* ΔA); dims = dims)
    return (C, ∂C)
end
```

We can now derive the reverse-mode rule.
The array form of the desired identity will be

```math
ds = \operatorname{real} \left( \sum_{ik}  \operatorname{conj}(\overline{C}_{i1k}) ~dC_{i1k} \right)
   = \operatorname{real} \left( \sum_{ijk} \operatorname{conj}(\overline{A}_{ijk}) ~dA_{ijk} \right)
```

We plug the differential identity into the middle expression to get

```math
\begin{align*}
    ds &= \operatorname{real} \left(\sum_{ijk}
                  \operatorname{conj}(\overline{C}_{i1k}) 2 \operatorname{real}(\operatorname{conj}(A_{ijk}) ~dA_{ijk})
              \right) \\
       &= \operatorname{real} \left( \sum_{ijk} 2 \operatorname{real}(\overline{C}_{i1k}) \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right).
\end{align*}
```

We now solve for $\overline{A}$:

```math
\overline{A}_{ijk} = \operatorname{conj}(2 \operatorname{real}( \overline{C}_{i1k} ) \operatorname{conj}(A_{ijk}))
                   = 2\operatorname{real}( \overline{C}_{i1k} ) A_{ijk}
```

Because none of this derivation really depended on the index (or indices), we can easily implement the `rrule` more generically using broadcasting:

```julia
function rrule(::typeof(sum), ::typeof(abs2), A::Array{<:RealOrComplex}; dims = :)
    function sum_abs2_pullback(ΔC)
        ∂abs2 = DoesNotExist()
        ∂A = @thunk(2 .* real.(ΔC) .* A)
        return (NO_FIELDS, ∂abs2, ∂A)
    end
    return sum(abs2, A; dims = dims), sum_abs2_pullback
end
```
