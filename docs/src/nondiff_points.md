# What to return for non-differentiable points
!!! info "What is the short version?"
    If the function is not differentiable due to e.g. a branch, like `abs`, your rule can reasonably claim the derivative at that point is the value from either branch, *or* any value in-between (e.g. for `abs` claiming 0 is a good idea).
    If it is not differentiable due to the primal not being defined on one side, you can set it to what ever you like.
    Your rule should claim a derivative that is *useful*.
In calculus one learns that if the derivative as computed by approaching from the left,
and the derivative one computes as approaching from the right are not equal then the derivative is not defined,
and we say the function is not differentiable at that point.
This is distinct from the notion captured by [`NoTangent`](@ref), which is that the tangent space itself is not defined: because in some sense the primal value can not be perturbed e.g. is is a discrete type.

However, contrary to what calculus says most autodiff systems will return an answer for such functions.
For example for: `abs_left(x) = (x <= 0) ? -x : x`, AD will say the derivative at `x=0` is `-1`.
Alternatively for:  `abs_right(x) = (x < 0) ? -x : x`, AD will say the derivative at `x=0` is `1`.
Those two examples are weird since they are equal at all points, but AD claims different derivatives at `x=0`.
The way to fix autodiff systems being weird is to write custom rules.
So what rule should we write for this case?

The obvious answer, would be to write a rule that throws an error if input at a point where calculus says the derivative is not defined.
Another option is to return some error signally value like `NaN`.
Which you *can* do.
However, this is not useful.
Instead we introduce what we call the **sub/super-differential convention**:

> It is always permissible to return any element of the sub/super-differential.
> You should return the most useful.

Or equivalently but considering the trivial singleton sub/super-differential seperately:

> At any point where the derivative is not defined, it is permissible to return any element of the sub/super-differential.
> You should return the most useful.

We will justify this further below, but first let us discuss what it means.

## What is the sub/super-differential?

The subderivative is defined only for locally convex functions, whereas the super-derivative is defined only for locally concave functions.
For our purpose we basically never care which we are working with and so write sub/super-derivative.

For a function $f$ at some point $x_0$ a sub/super-derivative is a real number $c$ such that there exists a open ball $\mathcal{B} \subset \mathrm{dom}(f)$ containing $x_0$,
and for all points $z \in \mathcal{B}$ the following holds:

$$\mathrm{sub -derivative:}\qquad f(z) - f(x_0) \ge c\,(z-x_0)$$

$$\mathrm{super-derivative:}\qquad f(z) - f(x_0) \le c\,(z-x_0)$$

We call the the set of all values $c$ the sub/super-differential at $x_0$.

More informally: consider a plot of the function.
The sub/super-differential at a point is the set of slopes of all lines you could draw touching that point, but with the lines either entirely above, or entirely below the curve.

It is best illustrated with a figure:

![plot showing subderiviatives](https://upload.wikimedia.org/wikipedia/commons/4/4e/Subderivative_illustration.png)

In this figure a plot of a function is shown in blue.
Two subtangent lines are plotted in red.
Their slopes are sub/super-derivatives at $x_0$, and they are two elements of the subdifferential.
If you flip it upside down, it would apply for the super-differential, with the lines being above the curve.

### Some key things to note.

For a function where the derivative is defined on both sides of the point:
 - the derivative on each side is a sub/super-derivative at the point
 - as is the mean of the derivative of each side
 - in-fact the mean of any subset (including the whole sub/super-differential) of sub/super-derivatives is also a sub/super-derivative.
 - if the derivative one one side is negative and the other positive then zero is a sub/super-derivative.

For a function that is only defined on one side of the point, the sub/super-differential is the full set of real numbers.
This by the subgradient convention leaves you free to chose *any* useful value.

!!! info "Does AD always return a sub/super-derivative? No"
    On consideration of this, one might be tempted to think that AD always returns a sub/super-derivative.
    And that in normal cases it return the only sub/super-derivative i.e. the actual derivative; and in other case it picks one of the branches.
    Thus all weirdness of AD disagreeing with calculus could be explained away in this way.
    **This is not the case.**
    As it is not necessarily true that all branches derivatives are correct sub/super-differentials for the function.
    Consider the function from [Abadi and Plotkin, 2019](https://dl.acm.org/doi/10.1145/3371106):
    `f(x::Float64) = x == 0.0 ? 0.0 : x`.
    The correct derivative of this function is `1` at all points, but most ADs will say that at `x=0` the derivative is zero.
    The fix for this, is to define a rule which *does* in fact return a sub/super-derivative.
    So one could say a correctly functioning AD with all needed rules does always return a sub/super-differential.
    

## What is useful ?

The sub/super-differential convention had two parts:
"It is always permissable to return any element of the sub/super-differential.
**You should return the most useful.**".
What is the most useful?
This is a value judgement you as a rule author will have to make.


### It is often zero

If zero is a sub/super-derivative, then it is often the most useful one.
Especially if the point is a local minima/maxima
For a function like `abs` or `x -> x<0 ? x : -2x` setting the non-differentiable point to either side would result in it leaving that maxima.

Further, a nice (albeit generally intractable) algorithm for finding the global optima is to take the set of all stationary points (i.e. points where the derivative is zero), combine that with the boundary points and evaluate the primal function at all of them.
The extrema of this set are the global optima.
This algorithm is only guaranteed correct for functions that are differentiable everywhere *or* that apply the sub/super-derivative convention and make all non-differentiable local optima claim to have a zero derivative.
It is also correct if you make other non-differentiable points have zero derivative, just slower.

### It is sometimes the non-zero, especially if it boarders a flat section

If a function has derivative zero in some large section of it's domain, like `relu` or `x->clamp(x, -1, 1)`,
a case can be made for choosing the non-zero derivative branch.
Depending exactly on the larger function being optimized, it is often the case that a zero gradient for this function means a total zero gradient (e.g. if the greater function is a chain of composed calls).
So once things move into the flat-region they stay there.
If we chose the other branch we move them into (just barely) the nonflat region.
If moving into the non-flat region was good they will move further there later.
If it is not good then they well be rapidly cast deeper into the flat region and we will not be at this boundary case.

### It is sometimes the mean
Either the mean of the branches, or the overall mean of the sub/super-differential (which may be equal).

A nice advantage of it is that it will agree with the result you will get from central finite differencing.
Which is what [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl) defaults to.

### All else being equal it is just one of the branches
Pick one.
It will be fast.
It will agree with either the forwards or reverse finite differencing results.

### It is almost never `Inf`, `NaN` or an error
Practically speaking, people are generally using AD to perform some gradient descent like optimization procedure.
`NaN` and `Inf` do not generally take us to nice places.
Erroring can be worse, especially if it is a local minima -- the optimization fully converges to that minima and then throws an error rather than reporting the result.
If the primal function takes a non-finite value or errors on one side, then we are in the case that we are at a boundary of the domain.
Which means we are free to chose *any* value.
In particular we often want to chose value of the derivative from **the other side where the primal is defined.**


## Why is the sub/super-differential convention permissible?

Math isn't real, and AD doesn't do calculus.
We are trying to accomplish some goal, and this approach works well.

One way to think about this  convention is to take an infinitely sharp discontinuity in a function and replace that discontinuty with a very (infinitesimally) small, smooth corner.
Due to the intermediate value theorem, we can then say that over this tiny interval, any derivative between the two extremes is realized and we are free to pick any one of them that we find useful as the 'canonical' value.

More specifically, consider our initial examples:
`abs_left(x) = (x <= 0) ? -x : x`, and `abs_right(x) = (x < 0) ? -x : x`.
These are a a primal level indistinguishable to the user.
It is impossible to tell which `Base.abs` uses without looking at the source.
Thus the rule author must be free to chose between assuming it is either.
Which is equivalent to saying they are free to chose to return the derivative or either branch.
We can then take the continuous relaxation of that choice: to chose any value between them.
Which for that case is choosing any sub-differential.
We then generalize from that chose into the sub/super-differential convention.

## How does this generalize to n-D
Carefully, but consistently.
