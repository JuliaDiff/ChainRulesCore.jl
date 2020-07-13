# Deriving Array Rules

One of the goals of the ChainRules interface is to make it easy to define your own rules for a function.
This tutorial attempts to demystify deriving and implementing custom rules for a large class of functions, with examples.

When the inputs and outputs are arrays and/or scalars, a modified version of Giles's method is a quick way of deriving [`frule`](@ref)s and [`rrule`](@ref)s.
Giles's method is succinctly explained in [^Giles2008] and its extended work [^Giles2008ext], but we will generalize it to handle arrays of arbitrary size with both real and complex entries.

Throughout this tutorial, we will ust the following type alias

```julia
const RealOrComplex = Union{Real,Complex}
```

## Forward-mode rules

### Approach

Consider a function

```julia
Ω = f(X::Array{<:RealOrComplex}...)::Array{<:RealOrComplex}
```

or in math notation

$$f: (\ldots, X_m, \ldots) \mapsto \Omega,$$

where the components of the arrays are written as $(X_m)_{i,\ldots,j}$ and $\Omega_{k,\ldots,l}$.
These variables are intermediates in a larger program (function) that, by considering only a single real input $t$ and real output $s$ can always be written as

$$t \mapsto (\ldots, X_m, \ldots) \mapsto \Omega \mapsto s,$$

where $t$ and $s$ are real numbers.
If we know the partial derivatives of $X_m$ with respect to $t$, $\frac{\partial X_m}{\partial t} = \dot{X}_m$, the chain rules gives the pushforward of $f$ as:

$$\dot{\Omega} = f_*(\ldots, \dot{X}_m, \ldots) =  \sum_m \sum_{i, \ldots, j} \frac{\partial \Omega}{\partial (X_m)_{i,\ldots,j}} (\dot{X}_m)_{i,\ldots,j}$$

That's ugly, but in practice we can often write it more simply by using forward mode rules for simpler functions, as we'll see below.
The main realization is that the forward-mode rules for arrays follow directly from the usual scalar chain rules.

### Matrix addition

```julia
Ω = A + B
```

This one is easy:

$$\Omega = A + B$$

$$\dot{\Omega} = \dot{A} + \dot{B}$$

We can implement the `frule` in ChainRules` notation:

```julia
function frule(
    (_, ΔA, ΔB),
    ::typeof(+), A::Array{<:RealOrComplex}, B::Array{<:RealOrComplex},
)
    Ω = A + B
    ∂Ω = ΔA + ΔB
    return (Ω, ∂Ω)
end
```

### Matrix multiplication

```julia
Ω = A * B
```

$$\Omega = A B$$

First we write in component form:

$$\Omega_{ij} = \sum_k A_{ik} B_{kj}$$

Then we use the product rule to get the scalar differential identity:

```math
\begin{align*}
\dot{\Omega}_{ij} &= \sum_k \left( \dot{A}_{ik} B_{kj} + A_{ik} \dot{B}_{kj} \right)
            && \text{apply scalar product rule } \frac{d}{dt}(x y) = \frac{dx}{dt} y + x \frac{dy}{dt} \\
        &= \sum_k \dot{A}_{ik} B_{kj} + \sum_k A_{ik} \dot{B}_{kj}
            && \text{split sum}
\end{align*}
```

But the last expression is just a sum of matrix products:

```math
\begin{equation}\label{diffprod}
\dot{\Omega} = \dot{A} B + A \dot{B}
\end{equation}
```

This is the matrix product rule, whose `frule` is

```julia
function frule(
    (_, ΔA, ΔB),
    ::typeof(*), A::Matrix{<:RealOrComplex}, B::Matrix{<:RealOrComplex},
)
    Ω = A * B
    ∂Ω = ΔA * B + A * ΔB
    return (Ω, ∂Ω)
end
```

### Matrix inversion

```julia
Ω = inv(A)
```

$$\Omega = A^{-1}$$

It's easiest to derive this rule from either constraint:

```math
\begin{align*}
\Omega A &= A^{-1} ~A = I\\
A \Omega &= A~ A^{-1} = I,
\end{align*}
```

where $I$ is the identity matrix.

We use the matrix product rule to differentiate the first constraint:

$$\dot{\Omega} A + \Omega \dot{A} = 0$$

We right-multiply both sides by $A^{-1}$ to isolate $\dot{\Omega}$:

```math
\begin{align}
0  &= \dot{\Omega}~ A~ A^{-1} + \Omega ~\dot{A}~ A^{-1} && \nonumber\\
   &= \dot{\Omega}~ I + \Omega ~\dot{A}~ A^{-1} && \text{apply } A~ A^{-1} = I \nonumber\\
   &= \dot{\Omega} + \Omega ~\dot{A}~ \Omega && \text{substitute } A^{-1} = \Omega \nonumber\\
\dot{\Omega} &= -\Omega ~\dot{A}~ \Omega && \text{solve for } \dot{\Omega} \label{invdiff}
\end{align}
```

We write the `frule` as

```julia
function frule((_, ΔA), ::typeof(inv), A::Matrix{<:RealOrComplex})
    Ω = inv(A)
    ∂Ω = -Ω * ΔA * Ω
    return (Ω, ∂Ω)
end
```

### Other useful identities

These identities are particularly useful:

```math
\begin{align*}
\frac{\partial}{\partial t} \left( \operatorname{real}(A) \right) &= \operatorname{real}(\dot{A})\\
\frac{\partial}{\partial t} \left( \operatorname{conj}(A) \right) &= \operatorname{conj}(\dot{A})\\
\frac{\partial}{\partial t} \left( A^T \right) &= \dot{A}^T\\
\frac{\partial}{\partial t} \left( A^H \right) &= \dot{A}^H\\
\frac{\partial}{\partial t} \left( \sum_{j}  A_{i \ldots j \ldots k} \right) &=
        \sum_{j} \dot{A}_{i \ldots j \ldots k},
\end{align*}
```

where $\cdot^H$ is the conjugate transpose (the `adjoint` function).

## Reverse-mode rules

Reverse-mode rules are a little more involved.
For a real scalar function $s = g(C)$, the differential of $s$ is the real part of the inner product (`LinearAlgebra.dot`) of the adjoint of $C$ (i.e. $\overline{C}$), and the differential of $C$:

```math
ds = \operatorname{real}\left( \langle \overline{C}, dC \rangle \right)
   = \operatorname{real}\left(
         \sum_{i,\dots,j} \operatorname{conj}(\overline{C}_{i,\dots,j}) ~dC_{i,\dots,j}
     \right),
```

where $\operatorname{conj}(\cdot)$ is the complex conjugate (`conj`), and $\operatorname{real}(\cdot)$ is the real part of its argument (`real`).

For matrices and vectors, we can write this as

$$ds = \operatorname{real}(\operatorname{tr}(\overline{C}^H dC)),$$

where $\operatorname{tr}$ is the matrix trace (`LinearAlgebra.tr`) function.

We can write the corresponding expression for $dA$ and $dB$ as

```math
\begin{align*}
ds &= \operatorname{real}\left( \operatorname{tr}\left( \overline{C}^H dC \right) \right) &&\\
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
    \overline{A} &= \left( \overline{A}^H \right)^H
                  = \left( \overline{C}^H \frac{\partial f}{\partial A} \right)^H
                  = \left( \frac{\partial f}{\partial A} \right)^H \overline{C}\\
    \overline{B} &= \left( \frac{\partial f}{\partial B} \right)^H \overline{C}
\end{align*}
```

Giles's method for deriving pullback functions is to first derive the differential identity (i.e. pushforward) using the above approach, then pre-multiply by $\overline{C}^H$, and take the real trace.
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
    Our method is identical to Giles's method, except we have replace the transpose with the conjugate transpose, and we have added the constraint that the inner product be real.
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
    \overline{A} &= \left( \overline{A}^H \right)^H
                  = \left( B \overline{C}^H \right)^H = \overline{C} B^H\\
    \overline{B} &= \left( \overline{B}^H \right)^H
                  = \left( \overline{C}^H A \right)^H = A^H \overline{C}
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

For more instructive examples of matrix and vector rules, see [^Giles2008ext] and the [LinearAlgebra rules in ChainRules](https://github.com/JuliaDiff/ChainRules.jl/tree/master/src/rulesets/LinearAlgebra).

## Generalizing to multidimensional arrays

For both forward- and reverse-mode rules for matrices, the first step was to write down the differential identities, which followed directly from the scalar differential identities.
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
    dC_{i1k} &= \sum_j \operatorname{real}\left( \operatorname{conj}(dA_{ijk})~ A_{ijk} +
                            \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right) \\
             &= \sum_j \operatorname{real}\left(
                    \operatorname{conj}\left(
                        \operatorname{conj}(A_{ijk}) ~dA_{ijk}
                    \right) +
                        \operatorname{conj}(A_{ijk}) ~dA_{ijk}
                \right\\
             &= \sum_j 2 \operatorname{real}\left( \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right)
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
                  \operatorname{conj}(\overline{C}_{i1k})
                  2 \operatorname{real}\left( \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right)
              \right) \\
       &= \operatorname{real} \left( \sum_{ijk}
              2 \operatorname{real}(\overline{C}_{i1k})
              \operatorname{conj}(A_{ijk}) ~dA_{ijk}
          \right).
\end{align*}
```

We now solve for $\overline{A}$:

```math
\overline{A}_{ijk} = \operatorname{conj}\left(
                         2 \operatorname{real}( \overline{C}_{i1k} )
                         \operatorname{conj}(A_{ijk})
                     \right)
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

## References

[^Giles2008]:
    > Giles M. B. Collected Matrix Derivative Results for Forward and Reverse Mode Algorithmic Differentiation.
    > In: Advances in Automatic Differentiation.
    > _Lecture Notes in Computational Science and Engineering_, vol 64: pp 35-44. Springer, Berlin (2008).
    > doi: [10.1007/978-3-540-68942-3_4](https://doi.org/10.1007/978-3-540-68942-3_4).
    > [pdf](https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf)

[^Giles2008ext]:
    > Giles M. B. An Extended Collection of Matrix Derivative Results for Forward and Reverse Mode Algorithmic Differentiation. (unpublished).
    > [pdf](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
