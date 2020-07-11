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

$$dC = \frac{\partial f}{\partial A} dA + \frac{\partial f}{\partial B} dB.$$

Notationally, we write the pushforward by replacing the differentials with the corresponding forward-mode sensitivities (e.g. replace $dC$ with $\dot{C}$):

$$\dot{C} = \frac{\partial f}{\partial A} \dot{A} + \frac{\partial f}{\partial B} \dot{B}.$$

The terms $\frac{\partial f}{\partial A}$ are array-array derivatives (i.e. a type of Jacobian).
We do not write these down explicitly, but we instead use differential identities to derive the terms $\frac{\partial f}{\partial A} dA$, which as we've seen behave like the Jacobian-vector-products $\frac{\partial f}{\partial A} \dot{A}$.
The differential identities follow directly from the usual scalar identities.

### Matrix addition

```julia
C = A + B
```

This one is easy:

$$C = A + B$$

$$dC = dA + dB$$

### Matrix multiplication

```julia
C = A * B
```

$$C = A B$$

First we write in component form:

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

Then we apply the product rule to get the scalar differential identity:

$$dC_{ij} = \sum_k dA_{ik}~ B_{kj} + A_{ik} ~dB_{kj}$$

But this is just a sum of matrix products:

$$dC = dA~ B + A ~dB$$

So we now have the matrix product rule.

### Matrix inversion

```julia
C = inv(A)
```

$$C = A^{-1}$$

It's easiest to derive this rule by writing the constraint:

$$C A = A C = I,$$

where $I$ is the identity matrix, and $C = A^{-1}$.

Now we use the matrix product rule to differentiate the constraint:

$$dC~ A + C ~dA = 0$$

We right-multiply both sides by $C = A^{-1}$ to isolate $dC$:

$$dC~ A C + C ~dA~ C = dC + C ~dA~ C = 0$$
$$dC = -C ~dA~ C$$

### Other useful identities

Two useful identities are $d(A^H) = dA^H$ and $d(A^T) = dA^T$, where $\cdot^H$ is the conjugate transpose (i.e. the `adjoint` function).

## Deriving reverse-mode rules

Reverse-mode rules are a little more involved.
For a real scalar function $s = g(C)$, the differential of $s$ is the real part of the Frobenius inner product (`LinearAlgebra.dot`) of the adjoint of $C$, $\overline{C}$, and the differential of $C$:

```math
ds = \Re\left( \langle \overline{C}, dC \rangle \right)
   = \Re\left( \sum_{i,\dots,j} \operatorname{conj}(\overline{C}_{i,\dots,j}) ~dC_{i,\dots,j} \right),
```

where $\operatorname{conj}(\cdot)$ is the complex conjugate (`conj`), and $\Re(x)$ is the real part of $x$ (i.e. `real(x)`).

For matrices and vectors, we can write this as

$$ds = \Re(\operatorname{tr}(\overline{C}^H dC)),$$

where $\operatorname{tr}$ is the matrix trace (`LinearAlgebra.tr`) function.

Plugging in the expression for $dC$, we get

```math
\begin{align*}
ds &= \Re\left( \operatorname{tr}\left(
          \overline{C}^H \frac{\partial f}{\partial A} dA +
          \overline{C}^H \frac{\partial f}{\partial B} dB
      \right) \right)\\
   &= \Re\left( \operatorname{tr}\left(
          \overline{C}^H \frac{\partial f}{\partial A} dA
      \right) \right) +
      \Re\left( \operatorname{tr}\left(
          \overline{C}^H \frac{\partial f}{\partial B} dB
      \right) \right)
\end{align*}
```

By applying the same definition of $ds$ to the intermediates $A$ and $B$, we get

```math
ds = \Re\left( \operatorname{tr}(\overline{A}^H  dA) \right) +
     \Re\left( \operatorname{tr}(\overline{B}^H ~dB) \right)
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

Giles' method for deriving pullback functions is to first derive the differential identity (i.e. pushforward) using the above approach, then pre-multiply by $\overline{C}^H$, and take the trace.
Subsequently, manipulate into this form and solve for the adjoint derivatives of the inputs.
Several properties of the trace function make this easier:

```math
\begin{align*}
    \operatorname{tr}(A+B) &= \operatorname{tr}(A) + \operatorname{tr}(B)\\
    \operatorname{tr}(A^T) &= \operatorname{tr}(A)\\
    \operatorname{tr}(A^H) &= \operatorname{conj}(\operatorname{tr}(A))\\
    \operatorname{tr}(AB) &= \operatorname{tr}(BA)
\end{align*}
```

!!! note
    Our method is identical to Giles' approach, except we have replace the transpose with the conjugate transpose, and we have added the constraint that the inner product be real.
    This produces the correct pullbacks for arrays with complex entries.

Here are a few examples.

### Matrix multiplication

```julia
C = A * B
```

We above derived

$$dC = dA~ B + A ~dB$$

We now multiply by $\overline{C}^H$ and take the real trace:

```math
\begin{align*}
ds &= \Re\left( \operatorname{tr}(\overline{C}^H ~dC) \right) \\
   &= \Re\left( \operatorname{tr}(\overline{C}^H ~dA~ B) \right) +
      \Re\left( \operatorname{tr}(\overline{C}^H  A ~dB) \right)
\end{align*}
```

We use the trace identities to manipulate the $dA$ expression:

```math
ds = \Re\left( \operatorname{tr}(B \overline{C}^H ~dA) \right) +
        \Re\left( \operatorname{tr}(\overline{C}^H A ~dB) \right)
```

That's it!
The expression is in the desired form to solve for the adjoints:

```math
\begin{align*}
    \overline{A} &= (B \overline{C}^H)^H = \overline{C} B^H\\
    \overline{B} &= (\overline{C}^H A)^H = A^H \overline{C}
\end{align*}
```

Using our notation conventions, we would implement the `rrule` as

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

$$C A = A C = I$$

Again, we derived the differential identity as

$$dC = -C ~dA~ C$$

Multiplying by $\overline{C}^H$ and taking the trace,

```math
    ds = \Re\left( \operatorname{tr}(\overline{C}^H ~dC) \right)
       = \Re\left( \operatorname{tr}(-\overline{C}^H C ~dA~ C) \right)
```

Applying the trace identities to manipulate into the desired form,

$$ds = \Re(\operatorname{tr}(-C \overline{C}^H C ~dA) ),$$

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

$$C_{i1k} = \sum_{j} |A_{ijk}|^2 = \sum_{j} \Re(\operatorname{conj}(A_{ijk}) A_{ijk})$$

The differential identity is

```math
\begin{align*}
    dC_{i1k} &= \sum_j \Re( \operatorname{conj}(dA_{ijk})~ A_{ijk} +
                            \operatorname{conj}(A_{ijk}) ~dA_{ijk} ) \\
             &= \sum_j \Re( \operatorname{conj}\left( \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right) +
                                                      \operatorname{conj}(A_{ijk}) ~dA_{ijk} )\\
             &= \sum_j 2 \Re( \operatorname{conj}(A_{ijk}) ~dA_{ijk} )\\
             &= 2 \Re\left( \sum_j \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right)\\
             &= 2 \Re \left( \langle A, dA \rangle \right),
\end{align*}
```

where we have used $(a + i b) + \operatorname{conj}(a + i b) = (a + i b) + (a - i b) = 2 a = 2 \Re (a + i b)$

We can then derive the reverse-mode rule.
The array form of the desired identity will be

```math
ds = \Re \left( \sum_{ik}  \operatorname{conj}(\overline{C}_{i1k}) ~dC_{i1k} \right)
   = \Re \left( \sum_{ijk} \operatorname{conj}(\overline{A}_{ijk}) ~dA_{ijk} \right)
```

We plug the differential identity into the middle expression to get

```math
\begin{align*}
    ds &= \Re \left(\sum_{ijk}
                  \operatorname{conj}(\overline{C}_{i1k}) 2 \Re(\operatorname{conj}(A_{ijk}) ~dA_{ijk})
              \right) \\
       &= \Re \left( \sum_{ijk} 2 \Re(\overline{C}_{i1k}) \operatorname{conj}(A_{ijk}) ~dA_{ijk} \right).
\end{align*}
```

We can now solve for $\overline{A}$:

```math
\overline{A}_{ijk} = \operatorname{conj}(2 \Re( \overline{C}_{i1k} ) \operatorname{conj}(A_{ijk}))
                   = 2\Re( \overline{C}_{i1k} ) A_{ijk}
```

Because none of this derivation really depended on the index (or indices), we can easily implement the `rrule` more generically using broadcasting:

```julia
function rrule(::typeof(sum), ::typeof(abs2), A::Array{<:RealOrComplex}; dims = :)
    function sum_abs2_pullback(ΔC)
        ∂abs2 = DoesNotExist()
        ∂A = @thunk(A .* 2 .* real.(ΔC))
        return (NO_FIELDS, ∂abs2, ∂A)
    end
    return sum(abs2, A; dims = dims), sum_abs2_pullback
end
```
