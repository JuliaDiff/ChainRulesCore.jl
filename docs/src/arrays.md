# Deriving Array Rules

One of the goals of the ChainRules interface is to make it easy to define your own rules for a function.
This tutorial attempts to demystify deriving and implementing custom rules for arrays with real and complex entries, with examples.
The approach we use is similar to the one succinctly explained and demonstrated in [^Giles2008] and its extended work [^Giles2008ext], but we generalize it to support functions of multidimensional arrays with both real and complex entries.

Throughout this tutorial, we will use the following type alias:

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

where the components of $X_m$ are written as $(X_m)_{i,\ldots,j}$.
The variables $X_m$ and $\Omega$ are intermediates in a larger program (function) that, by considering only a single real input $t$ and real output $s$ can always be written as

$$t \mapsto (\ldots, X_m, \ldots) \mapsto \Omega \mapsto s,$$

where $t$ and $s$ are real numbers.
If we know the partial derivatives of $X_m$ with respect to $t$, $\frac{dX_m}{dt} = \dot{X}_m$, the chain rule gives the pushforward of $f$ as:

```math
\begin{equation} \label{pf}
\dot{\Omega}
    = f_*(\ldots, \dot{X}_m, \ldots)
    = \sum_m \sum_{i, \ldots, j}
        \frac{\partial \Omega}{ \partial (X_m)_{i,\ldots,j} } (\dot{X}_m)_{i,\ldots,j}
\end{equation}
```

That's ugly, but in practice we can often write it more simply by using forward mode rules for simpler functions, as we'll see below.
The forward-mode rules for arrays follow directly from the usual scalar chain rules.

### Array addition

```julia
Ω = A + B
```

This one is easy:

$$\Omega = A + B$$

$$\dot{\Omega} = \dot{A} + \dot{B}$$

We can implement the `frule` in ChainRules's notation:

```julia
function frule(
    (_, ΔA, ΔB),
    ::typeof(+),
    A::Array{<:RealOrComplex},
    B::Array{<:RealOrComplex},
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

Then we use the product rule to get the pushforward for each scalar entry:

```math
\begin{align*}
\dot{\Omega}_{ij}
    &= \sum_k \left( \dot{A}_{ik} B_{kj} + A_{ik} \dot{B}_{kj} \right)
        && \text{apply scalar product rule }
            \frac{d}{dt}(x y) = \frac{dx}{dt} y + x \frac{dy}{dt} \\
    &= \sum_k \dot{A}_{ik} B_{kj} + \sum_k A_{ik} \dot{B}_{kj}
        && \text{split sum}
\end{align*}
```

But the last expression is just the component form of a sum of matrix products:

```math
\begin{equation}\label{diffprod}
\dot{\Omega} = \dot{A} B + A \dot{B}
\end{equation}
```

This is the matrix product rule, and we write its `frule` as

```julia
function frule(
    (_, ΔA, ΔB),
    ::typeof(*),
    A::Matrix{<:RealOrComplex},
    B::Matrix{<:RealOrComplex},
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

It's easiest to derive this rule from either of the two constraints:

```math
\begin{align*}
\Omega A &= A^{-1} ~A = I\\
A \Omega &= A~ A^{-1} = I,
\end{align*}
```

where $I$ is the identity matrix.

We use the matrix product rule to differentiate the first constraint:

$$\dot{\Omega} A + \Omega \dot{A} = 0$$

Then, right-multiply both sides by $A^{-1}$ to isolate $\dot{\Omega}$:

```math
\begin{align}
0  &= \dot{\Omega}~ A~ A^{-1} + \Omega ~\dot{A}~ A^{-1} \nonumber\\
   &= \dot{\Omega}~ I + \Omega ~\dot{A}~ A^{-1}
       && \text{use } A~ A^{-1} = I \nonumber\\
   &= \dot{\Omega} + \Omega \dot{A} \Omega
       && \text{substitute } A^{-1} = \Omega \nonumber\\
\dot{\Omega}
   &= -\Omega \dot{A} \Omega
       && \text{solve for } \dot{\Omega} \label{invdiff}
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
\frac{d}{dt} \left( \operatorname{real}(A) \right) &= \operatorname{real}(\dot{A})\\
\frac{d}{dt} \left( \operatorname{conj}(A) \right) &= \operatorname{conj}(\dot{A})\\
\frac{d}{dt} \left( A^T \right) &= \dot{A}^T\\
\frac{d}{dt} \left( A^H \right) &= \dot{A}^H\\
\frac{d}{dt} \left( \sum_{j}  A_{i \ldots j \ldots k} \right) &=
    \sum_{j} \dot{A}_{i \ldots j \ldots k},
\end{align*}
```

where $\cdot^H = \operatorname{conj}(\cdot^T)$ is the conjugate transpose (the `adjoint` function).

## Reverse-mode rules

### Approach

Reverse-mode rules are a little less intuitive, but we can re-use our pushforwards to simplify their derivation.
Recall our program:

$$t \mapsto (\ldots, X_m, \ldots) \mapsto \Omega \mapsto s,$$

At any step in the program, if we have intermediates $X_m$, we can write down the derivative $\frac{ds}{dt}$ in terms of the tangents $\dot{X}_m = \frac{dX_m}{dt}$ and adjoints $\overline{X}_m = \frac{\partial s}{\partial X_m}$

```math
\begin{align*}
\frac{ds}{dt}
    &= \sum_m \operatorname{real}\left( \sum_{i,\ldots,j}
           \operatorname{conj}\left( \frac{\partial s}{\partial (X_m)_{i,\ldots,j}} \right)
           \frac{d (X_m)_{i,\ldots,j}}{dt}
       \right)\\
    &= \sum_m \operatorname{real}\left( \sum_{i,\ldots,j}
           \operatorname{conj} \left( (\overline{X}_m)_{i,\ldots,j} \right)
           (\dot{X}_m)_{i,\ldots,j}
       \right)\\
    &= \sum_m \operatorname{real}\left( \operatorname{dot}\left(
           \overline{X}_m, \dot{X}_m
       \right) \right),
\end{align*}
```

where $\operatorname{conj}(\cdot)$ is the complex conjugate (`conj`), $\operatorname{real}(\cdot)$ is the real part of its argument (`real`), and $\operatorname{dot}(\cdot, \cdot)$ is the inner product (`LinearAlgebra.dot`).
Because this equation follows at any step of the program, we can equivalently write 

```math
\frac{ds}{dt} = \operatorname{real}\left( \operatorname{dot}\left(
                    \overline{\Omega}, \dot{\Omega}
                \right) \right),
```

which gives the identity

```math
\begin{equation} \label{pbident}
\operatorname{real}\left( \operatorname{dot}\left(
    \overline{\Omega}, \dot{\Omega}
\right) \right) = 
\sum_m \operatorname{real}\left( \operatorname{dot}\left(
    \overline{X}_m, \dot{X}_m
\right) \right).
\end{equation}
```

For matrices and vectors, $\operatorname{dot}(A, B) = \operatorname{tr}(A^H B)$, and the identity simplifies to:

```math
\begin{equation} \label{pbidentmat}
\operatorname{real}\left( \operatorname{tr}\left(
    \overline{\Omega}^H \dot{\Omega}
\right) \right) =
\sum_m \operatorname{real} \left( \operatorname{tr} \left(
    \overline{X}_m^H \dot{X}_m
\right) \right),
\end{equation}
```

where $\operatorname{tr}(\cdot)$ is the matrix trace (`LinearAlgebra.tr`) function.

Our approach for deriving the adjoints $\overline{X}_m$ is then:

1. Derive the pushforward ($\dot{\Omega}$ in terms of $\dot{X}_m$) using \eqref{pf}.
2. Substitute this expression for $\dot{\Omega}$ into the left-hand side of \eqref{pbident}.
3. Manipulate until it looks like the right-hand side of \eqref{pbident}.
4. Solve for each $\overline{X}_m$.

Note that the final expressions for the adjoints will not contain any $\dot{X}_m$ terms.

!!! note
    Why do we conjugate, and why do we only use the real part of the dot product in \eqref{pbident}?
    Recall from [Complex Numbers](complex.md) that we treat a complex number as a pair of real numbers.
    These identities are a direct consequence of this convention.
    Consider $\frac{ds}{dt}$ for a scalar function $f: (x + i y) \mapsto (u + i v)$:
    ```math
    \begin{align*}
    \frac{ds}{dt}
        &= \operatorname{real}\left( \operatorname{dot}\left(
               \overline{x} + i \overline{y}, \dot{x} + i \dot{y}
           \right) \right) \\
        &= \operatorname{real}\left(
               \operatorname{conj} \left( \overline{x} + i \overline{y} \right)
               \left( \dot{x} + i \dot{y} \right)
           \right) \\
        &= \operatorname{real}\left(
               \left( \overline{x} - i \overline{y} \right)
               \left( \dot{x} + i \dot{y} \right)
           \right) \\
        &= \operatorname{real}\left(
               \left( \overline{x} \dot{x} + \overline{y} \dot{y} \right) +
               i \left( \overline{x} \dot{y} - \overline{y} \dot{x} \right)
           \right)\\
        &= \overline{x} \dot{x} + \overline{y} \dot{y}\\
    \end{align*}
    ```
    which is exactly what the identity would produce if we had written the function as $f: (x, y) \mapsto (u, v)$.

For matrices and vectors, several properties of the trace function come in handy:

```math
\begin{align}
\operatorname{tr}(A+B) &= \operatorname{tr}(A) + \operatorname{tr}(B) \label{trexpand}\\
\operatorname{tr}(A^T) &= \operatorname{tr}(A) \nonumber\\
\operatorname{tr}(A^H) &= \operatorname{conj}(\operatorname{tr}(A)) \nonumber\\
\operatorname{tr}(AB) &= \operatorname{tr}(BA) \label{trperm}
\end{align}
```

Now let's derive a few pullbacks using this approach.

### Matrix multiplication

```julia
Ω = A * B
```

We above derived in \eqref{diffprod} the pushforward

$$\dot{\Omega} = \dot{A} B + A \dot{B}$$

Using \eqref{pbidentmat}, we now multiply by $\overline{\Omega}^H$ and take the real trace:

```math
\begin{align*}
\operatorname{real}\left( \operatorname{tr} \left(
        \overline{\Omega}^H \dot{\Omega}
\right) \right)
    &= \operatorname{real}\left( \operatorname{tr} \left( \overline{\Omega}^H ~\left(
           \dot{A} B + A \dot{B}
       \right) \right) \right)
           && \text{substitute } \dot{\Omega} \text{ from } \eqref{diffprod}\\
    &= \operatorname{real}\left( \operatorname{tr} \left(
           \overline{\Omega}^H \dot{A} B
       \right) \right) +
       \operatorname{real}\left( \operatorname{tr} \left(
           \overline{\Omega}^H  A \dot{B}
       \right) \right)
           && \text{expand using } \eqref{trexpand} \\
    &= \operatorname{real}\left( \operatorname{tr} \left(
           B \overline{\Omega}^H \dot{A}
       \right) \right) +
       \operatorname{real}\left( \operatorname{tr} \left(
           \overline{\Omega}^H A \dot{B}
       \right) \right)
           && \text{rearrange the left term using } \eqref{trperm}\\
    &= \operatorname{real}\left( \operatorname{tr} \left(
           \overline{A}^H  \dot{A}
       \right) \right) +
       \operatorname{real}\left( \operatorname{tr} \left(
           \overline{B}^H \dot{B}
       \right) \right)
           && \text{right-hand side of } \eqref{pbidentmat}
\end{align*}
```

That's it!
The expression is in the desired form to solve for the adjoints by comparing the last two lines:

```math
\begin{align*}
B \overline{\Omega}^H \dot{A} &= \overline{A}^H  \dot{A}, \quad
    && \overline{A} = \overline{\Omega} B^H\\
\overline{\Omega}^H A \dot{B} &= \overline{B}^H \dot{B}, \quad
    && \overline{B} = A^H \overline{\Omega}
\end{align*}
```

Using ChainRules's notation, we would implement the `rrule` as

```julia
function rrule(::typeof(*), A::Matrix{<:RealOrComplex}, B::Matrix{<:RealOrComplex})
    function times_pullback(ΔΩ)
        ∂A = @thunk(ΔΩ * B')
        ∂B = @thunk(A' * ΔΩ)
        return (NO_FIELDS, ∂A, ∂B)
    end
    return A * B, times_pullback
end
```

### Matrix inversion

```julia
Ω = inv(A)
```

In \eqref{invdiff}, we derived the pushforward as

$$\dot{\Omega} = -\Omega \dot{A} \Omega$$

Using \eqref{pbidentmat},

```math
\begin{align*}
\operatorname{real}\left( \operatorname{tr} \left(
    \overline{\Omega}^H \dot{\Omega}
\right) \right)
    &= \operatorname{real}\left( \operatorname{tr} \left(
           -\overline{\Omega}^H \Omega \dot{A} \Omega
       \right) \right)
           && \text{substitute } \eqref{invdiff}\\
    &= \operatorname{real}\left( \operatorname{tr} \left(
           -\Omega \overline{\Omega}^H \Omega \dot{A}
       \right) \right)
           && \text{rearrange using } \eqref{trperm}\\
    &= \operatorname{real}\left( \operatorname{tr} \left(
           \overline{A}^H  \dot{A}
       \right) \right)
           && \text{right-hand side of } \eqref{pbidentmat}
\end{align*}
```

we can now solve for $\overline{A}$:

```math
\overline{A} = \left( -\Omega \overline{\Omega}^H \Omega \right)^H
             = -\Omega^H \overline{\Omega} \Omega^H
```

We can implement the resulting `rrule` as

```julia
function rrule(::typeof(inv), A::Matrix{<:RealOrComplex})
    Ω = inv(A)
    function inv_pullback(ΔΩ)
        ∂A = -Ω' * ΔΩ * Ω'
        return (NO_FIELDS, ∂A)
    end
    return Ω, inv_pullback
end
```

## A multidimensional array example

We presented the approach for deriving pushforwards and pullbacks for arrays of arbitrary dimensions, so let's cover an example.
For multidimensional arrays, it's often easier to work in component form.
Consider the following function:

```julia
Ω = sum(abs2, X::Array{<:RealOrComplex,3}; dims=2)::Array{<:Real,3}
```

which we write as

```math
\Omega_{i1k} = \sum_{j} |X_{ijk}|^2
             = \sum_{j} \operatorname{real} \left(
                  \operatorname{conj} \left( X_{ijk} \right) X_{ijk}
               \right)
```

The pushforward from \eqref{pf} is

```math
\begin{align}
\dot{\Omega}_{i1k}
    &= \sum_j \operatorname{real}\left(
           \operatorname{conj} \left( \dot{X}_{ijk} \right) X_{ijk} +
           \operatorname{conj} \left( X_{ijk} \right) \dot{X}_{ijk} \right) \nonumber\\
    &= \sum_j \operatorname{real}\left(
            \operatorname{conj}\left(
                \operatorname{conj} \left( X_{ijk} \right) \dot{X}_{ijk}
            \right) +
            \operatorname{conj}(X_{ijk}) \dot{X}_{ijk}
       \right) \nonumber\\
    &= \sum_j 2 \operatorname{real}\left(
           \operatorname{conj} \left( X_{ijk} \right) \dot{X}_{ijk}
       \right), \label{sumabspf}
\end{align}
```

where in the last step we have used the fact that for all real $a$ and $b$,

```math
(a + i b) + \operatorname{conj}(a + i b)
    = (a + i b) + (a - i b)
    = 2 a
    = 2 \operatorname{real} (a + i b).
```

Because none of this derivation depended on the index (or indices), we implement `frule` generically as

```julia
function frule(
    (_, _, ΔX),
    ::typeof(sum),
    ::typeof(abs2),
    X::Array{<:RealOrComplex};
    dims = :,
)
    Ω = sum(abs2, X; dims = dims)
    ∂Ω = sum(2 .* real.(conj.(X) .* ΔX); dims = dims)
    return (Ω, ∂Ω)
end
```

We can now derive the reverse-mode rule.
The array form of \eqref{pbident} is

```math
\begin{align*}
\operatorname{real}\left( \operatorname{dot}\left(
    \overline{\Omega}, \dot{\Omega}
\right) \right)
    &= \operatorname{real} \left( \sum_{ik}
           \operatorname{conj} \left( \overline{\Omega}_{i1k} \right) \dot{\Omega}_{i1k}
       \right)
           && \text{expand left-hand side of } \eqref{pbident}\\
    &= \operatorname{real} \left(\sum_{ijk}
           \operatorname{conj} \left( \overline{\Omega}_{i1k} \right)
           2 \operatorname{real}\left(
               \operatorname{conj} \left( X_{ijk} \right) \dot{X}_{ijk}
           \right)
       \right)
           && \text{substitute } \eqref{sumabspf}\\
    &= \operatorname{real} \left( \sum_{ijk}
           \left(
               2 \operatorname{real} \left( \overline{\Omega}_{i1k} \right)
               \operatorname{conj} \left( X_{ijk} \right)
           \right) \dot{X}_{ijk}
       \right)
           && \text{bring } \dot{X}_{ijk} \text{ outside of } \operatorname{real}\\
    &= \operatorname{real} \left( \sum_{ijk}
           \operatorname{conj} \left( \overline{X}_{ijk} \right) \dot{X}_{i1k}
       \right)
           && \text{expand right-hand side of } \eqref{pbident}
\end{align*}
```

We now solve for $\overline{X}$:

```math
\begin{align*}
\overline{X}_{ijk}
    &= \operatorname{conj}\left(
            2 \operatorname{real} \left( \overline{\Omega}_{i1k} \right)
            \operatorname{conj} \left( X_{ijk} \right)
        \right)\\
    &= 2\operatorname{real} \left( \overline{\Omega}_{i1k} \right) X_{ijk}
\end{align*}
```

Like the `frule`, this `rrule` can be implemented generically:

```julia
function rrule(::typeof(sum), ::typeof(abs2), X::Array{<:RealOrComplex}; dims = :)
    function sum_abs2_pullback(ΔΩ)
        ∂abs2 = DoesNotExist()
        ∂X = @thunk(2 .* real.(ΔΩ) .* X)
        return (NO_FIELDS, ∂abs2, ∂X)
    end
    return sum(abs2, X; dims = dims), sum_abs2_pullback
end
```

## Functions that return a tuple

Every Julia function returns a single output.
For example, let's look at `LinearAlgebra.logabsdet`, the logarithm of the absolute value of the determinant of a matrix, which returns $\log |\det(A)|$ and $\operatorname{sign}(\det A) = \frac{\det A}{| \det A |}$:

```julia
(l, s) = logabsdet(A)
```

The return type is actually a single output, a tuple of scalars, but when deriving, we treat them as multiple outputs.
The left-hand side of \eqref{pbident} then becomes a sum over terms, just like the right-hand side.

Let's derive the forward- and reverse-mode rules for `logabsdet`.

```math
\begin{align*}
l &= \log |\det(A)|\\
s &= \operatorname{sign}(\det(A)),
\end{align*}
```

where $\operatorname{sign}(x) = \frac{x}{|x|}$.

### Forward-mode rule

To make this easier, let's break the computation into more manageable steps:

```math
\begin{align*}
d &= \det(A)\\
a &= |d| = \sqrt{\operatorname{real} \left( \operatorname{conj}(d) d \right)}\\
l &= \log a\\
s &= \frac{d}{a}
\end{align*}
```

We'll make frequent use of the identities:

$$d = a s$$
$$\operatorname{conj}(s) s = \frac{\operatorname{conj}(d) d}{a^2} = \frac{a^2}{a^2} = 1$$

It will also be useful to define $b = \operatorname{tr}\left( A^{-1} \dot{A} \right)$.

For $\dot{d}$, we use the pushforward for the determinant given in section 2.2.4 of [^Giles2008ext]:

$$\dot{d} = d b$$

Now we'll compute the pushforwards for the remaining steps.

```math
\begin{align*}
\dot{a} &= \frac{1}{2 a} \frac{d}{dt}
                         \operatorname{real}\left( \operatorname{conj}(d) d \right)\\
        &= \frac{2}{2 a} \operatorname{real} \left( \operatorname{conj}(d) \dot{d} \right)\\
        &= \operatorname{real} \left( \operatorname{conj}(s) \dot{d} \right)
            && \text{use } d = a s \\
        &= \operatorname{real} \left( \operatorname{conj}(s) d b \right)
            && \text{substitute } \dot{d} \\
\dot{l} &= a^{-1} \dot{a}\\
        &= a^{-1} \operatorname{real} \left( \operatorname{conj}(s) d b \right)
            && \text{substitute } \dot{a}\\
        &= \operatorname{real} \left( \operatorname{conj}(s) s b \right)
            && \text{use } d = a s \\
        &= \operatorname{real} \left(b \right)
            && \text{use } \operatorname{conj}(s) s = 1\\
\dot{s} &= a^{-1} \dot{d} - a^{-2} d \dot{a}\\
        &= a^{-1} \left( \dot{d} - \dot{a} s \right)
            && \text{use } d = a s \\
        &= a^{-1} \left(
               \dot{d} - \operatorname{real} \left( \operatorname{conj}(s) \dot{d} \right) s
           \right)
            && \text{substitute } \dot{a}\\
        &= a^{-1} \left(
               \dot{d} - \left(
                   \operatorname{conj}(s) \dot{d} -
                   i \operatorname{imag} \left( \operatorname{conj}(s) \dot{d} \right)
               \right) s
           \right)
            && \text{use } \operatorname{real}(x) = x - i \operatorname{imag}(x)\\
        &= a^{-1} \left(
               \dot{d} - \left( \operatorname{conj}(s) s \right) \dot{d} +
               i \operatorname{imag} \left( \operatorname{conj}(s) \dot{d} \right) s 
               \right)\\
        &= i a^{-1} \operatorname{imag} \left( \operatorname{conj}(s) \dot{d} \right) s
            && \text{use } \operatorname{conj}(s) s = 1\\
        &= i a^{-1} \operatorname{imag} \left( \operatorname{conj}(s) d b \right) s
            && \text{substitute } \dot{d}\\
        &= i \operatorname{imag} \left( \operatorname{conj}(s) s b \right) s
            && \text{use } d = a s \\
        &= i \operatorname{imag}(b) s
            && \text{use } \operatorname{conj}(s) s = 1
\end{align*}
```

Note that the term $b$ is reused.
In summary, after all of that work, the final pushforward is quite simple:

```math
\begin{align}
b &= \operatorname{tr} \left( A^{-1} \dot{A} \right) \label{logabsdet_b} \\
\dot{l} &= \operatorname{real}(b) \label{logabsdet_ldot}\\
\dot{s} &= i \operatorname{imag}(b) s \label{logabsdet_sdot}\\
\end{align}
```

We can define the `frule` as:

```julia
function frule((_, ΔA), ::typeof(logabsdet), A::Matrix{<:RealOrComplex})
    # The primal function uses the lu decomposition to compute logabsdet
    # we reuse this decomposition to compute inv(A) * ΔA
    F = lu(A, check = false)
    Ω = logabsdet(F)  # == logabsdet(A)
    b = tr(F \ ΔA)  # == tr(inv(A) * ΔA)
    s = last(Ω)
    ∂l = real(b)
    # for real A, ∂s will always be zero (because imag(b) = 0)
    # this is type-stable because the eltype is known
    ∂s = eltype(A) <: Real ? Zero() : im * imag(b) * s
    # tangents of tuples are of type Composite{<:Tuple}
    ∂Ω = Composite{typeof(Ω)}(∂l, ∂s)
    return (Ω, ∂Ω)
end
```

### Reverse-mode rule

```math
\begin{align*}
&\operatorname{real}\left( \operatorname{tr}\left(
    \overline{l}^H \dot{l}
\right) \right) +
\operatorname{real}\left( \operatorname{tr}\left(
    \overline{s}^H \dot{s}
\right) \right)
    && \text{left-hand side of } \eqref{pbidentmat}\\
&= \operatorname{real}\left( 
       \operatorname{conj} \left( \overline{l} \right) \dot{l} +
       \operatorname{conj} \left( \overline{s} \right) \dot{s}
   \right) \\
&= \operatorname{real}\left( 
       \operatorname{conj} \left( \overline{l} \right) \operatorname{real}(b) +
       i \operatorname{conj} \left( \overline{s} \right) s \operatorname{imag}(b)
   \right)
       && \text{substitute } \eqref{logabsdet_ldot} \text{ and } \eqref{logabsdet_sdot} \\
&= \operatorname{real}\left( 
       \operatorname{real}\left( \overline{l} \right) \operatorname{real}(b) -
       \operatorname{imag} \left(
           \operatorname{conj} \left( \overline{s} \right) s
       \right) \operatorname{imag}(b)
   \right)
       && \text{discard imaginary parts} \\
&= \operatorname{real}\left(
       \left(
           \operatorname{real} \left( \overline{l} \right) +
           i \operatorname{imag} \left(
               \operatorname{conj} \left( \overline{s} \right) s
           \right)
       \right) b
   \right)
       && \text{gather parts of } b \\
&= \operatorname{real}\left(
       \left(
           \operatorname{real} \left( \overline{l} \right) +
           i \operatorname{imag} \left(
               \operatorname{conj} \left( \overline{s} \right) s
           \right)
       \right)
       \operatorname{tr}(A^{-1} \dot{A})
   \right)
       && \text{substitute } b \text{ from } \eqref{logabsdet_b} \\
&= \operatorname{real}\left( \operatorname{tr} \left(
       \left(
           \operatorname{real} \left( \overline{l} \right) +
           i \operatorname{imag} \left(
               \operatorname{conj} \left( \overline{s} \right) s
           \right)
       \right)
       A^{-1} \dot{A}
   \right) \right)
       && \text{bring scalar within } \operatorname{tr} \\
&= \operatorname{real}\left( \operatorname{tr} \left( \overline{A}^H \dot{A} \right) \right)
       && \text{right-hand side of } \eqref{pbidentmat}\\
\end{align*}
```

Now we solve for $\overline{A}$:

```math
\begin{align*}
\overline{A} &= \left( \left(
    \operatorname{real} \left( \overline{l} \right) +
    i \operatorname{imag} \left( \operatorname{conj} \left( \overline{s} \right) s \right)
\right) A^{-1} \right)^H\\
&= \left(
    \operatorname{real} \left( \overline{l} \right) +
    i \operatorname{imag} \left( \operatorname{conj} \left( s \right) \overline{s} \right)
\right) A^{-H}
\end{align*}
```

The `rrule` can be implemented as

```julia
function rrule(::typeof(logabsdet), A::Matrix{<:RealOrComplex})
    # The primal function uses the lu decomposition to compute logabsdet
    # we reuse this decomposition to compute inv(A)
    F = lu(A, check = false)
    Ω = logabsdet(F)  # == logabsdet(A)
    s = last(Ω)
    function logabsdet_pullback(ΔΩ)
        (Δl, Δs) = ΔΩ
        f = conj(s) * Δs
        imagf = f - real(f)  # 0 for real A and Δs, im * imag(f) for complex A and/or Δs
        g = real(Δl) + imagf
        ∂A = g * inv(F)'  # == g * inv(A)'
        return (NO_FIELDS, ∂A)
    end
    return (Ω, logabsdet_pullback)
end
```

!!! note
    It's a good idea when deriving pushforwards and pullbacks to verify that they make sense.
    For the pushforward, since $l$ is real, it follows that $\dot{l}$ is too.

    What about $\dot{s}$?
    Well, $s = \frac{d}{|d|}$ is point on the unit circle in the complex plane.
    Multiplying a complex number by $i$ rotates it counter-clockwise by 90°.
    So the expression for $\dot{s}$ takes a real number, $\operatorname{imag}(b)$, multiplies by $s$ to make it parallel to $s$, then multiplies by $i$ to make it perpendicular to $s$, that is, perfectly tangent to the unit complex circle at $s$.

    For the pullback, it again follows that only the real part of $\overline{l}$ is pulled back.

    ``\operatorname{conj}(s)`` rotates a number parallel to $s$ to the real line.
    So $\operatorname{conj}(s) \overline{s}$ rotates $\overline{s}$ so that its imaginary part is the part that was tangent to the complex circle at $s$, while the real part is the part that was not tangent.
    Then the pullback isolates the imaginary part, which effectively is a projection.
    That is, any part of the adjoint $\overline{s}$ that is not tangent to the complex circle at $s$ will not contribute to $\overline{A}$.

## More examples

For more instructive examples of array rules, see [^Giles2008ext] (real vector and matrix rules) and the [LinearAlgebra rules in ChainRules](https://github.com/JuliaDiff/ChainRules.jl/tree/master/src/rulesets/LinearAlgebra).

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
