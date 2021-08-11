# [How do chain rules work for complex functions?](@id complexfunctions)

ChainRules follows the convention that `frule` applied to a function ``f(x + i y) = u(x,y) + i v(x,y)`` with perturbation ``\Delta x + i \Delta y`` returns the value and
```math
\tfrac{\partial u}{\partial x} \, \Delta x + \tfrac{\partial u}{\partial y} \, \Delta y + i \, \Bigl( \tfrac{\partial v}{\partial x} \, \Delta x + \tfrac{\partial v}{\partial y} \, \Delta y \Bigr)
.
```
Similarly, `rrule` applied to the same function returns the value and a pullback function which, when applied to the adjoint ``\Delta u + i \Delta v``, returns
```math
\Delta u \, \tfrac{\partial u}{\partial x} + \Delta v \, \tfrac{\partial v}{\partial x} + i \, \Bigl(\Delta u \, \tfrac{\partial u }{\partial y} + \Delta v \, \tfrac{\partial v}{\partial y} \Bigr)
.
```
If we interpret complex numbers as vectors in ``\mathbb{R}^2``, then `frule` (`rrule`) corresponds to multiplication with the (transposed) Jacobian of ``f(z)``, i.e. `frule` corresponds to
```math
\begin{pmatrix}
\tfrac{\partial u}{\partial x} \, \Delta x + \tfrac{\partial u}{\partial y} \, \Delta y
\\
\tfrac{\partial v}{\partial x} \, \Delta x + \tfrac{\partial v}{\partial y} \, \Delta y
\end{pmatrix}
=
\begin{pmatrix}
\tfrac{\partial u}{\partial x} & \tfrac{\partial u}{\partial y} \\
\tfrac{\partial v}{\partial x} & \tfrac{\partial v}{\partial y} \\
\end{pmatrix}
\begin{pmatrix}
\Delta x \\ \Delta y
\end{pmatrix}

```
and `rrule` corresponds to
```math
\begin{pmatrix}
\tfrac{\partial u}{\partial x} \, \Delta u + \tfrac{\partial v}{\partial x} \, \Delta v
\\
\tfrac{\partial u}{\partial y} \, \Delta u + \tfrac{\partial v}{\partial y} \, \Delta v
\end{pmatrix}
=
\begin{pmatrix}
\tfrac{\partial u}{\partial x} & \tfrac{\partial u}{\partial y} \\
\tfrac{\partial v}{\partial x} & \tfrac{\partial v}{\partial y} \\
\end{pmatrix}^\mathsf{T}
\begin{pmatrix}
\Delta u \\ \Delta v
\end{pmatrix}
.
```
The Jacobian of ``f:\mathbb{C} \to \mathbb{C}`` interpreted as a function ``\mathbb{R}^2 \to \mathbb{R}^2`` can hence be evaluated using either of the following functions.
```julia
function jacobian_via_frule(f,z)
    du_dx, dv_dx = reim(frule((ZeroTangent(), 1),f,z)[2])
    du_dy, dv_dy = reim(frule((ZeroTangent(),im),f,z)[2])
    return [
        du_dx  du_dy
        dv_dx  dv_dy
    ]
end
```
```julia
function jacobian_via_rrule(f,z)
    _, pullback = rrule(f,z)
    du_dx, du_dy = reim(pullback( 1)[2])
    dv_dx, dv_dy = reim(pullback(im)[2])
    return [
        du_dx  du_dy
        dv_dx  dv_dy
    ]
end
```

If ``f(z)`` is holomorphic, then the derivative part of `frule` can be implemented as ``f'(z) \, \Delta z`` and the derivative part of `rrule` can be implemented as ``\bigl(f'(z)\bigr)^* \, \Delta f``, where ``\cdot^*`` is the complex conjugate.
Consequently, holomorphic derivatives can be evaluated using either of the following functions.
```julia
function holomorphic_derivative_via_frule(f,z)
    fz,df_dz = frule((ZeroTangent(),1),f,z)
    return df_dz
end
```
```julia
function holomorphic_derivative_via_rrule(f,z)
    fz, pullback = rrule(f,z)
    dself, conj_df_dz = pullback(1)
    return conj(conj_df_dz)
end
```

!!! note
    There are various notions of complex derivatives (holomorphic and Wirtinger derivatives, Jacobians, gradients, etc.) which differ in subtle but important ways.
    The goal of ChainRules is to provide the basic differentiation rules upon which these derivatives can be implemented, but it does not implement these derivatives itself.
    It is recommended that you carefully check how the above definitions of `frule` and `rrule` translate into your specific notion of complex derivative, since getting this wrong will quietly give you wrong results.
