# What to return for non-differentiable points
!!! info "What is the short version?"
    If the function is not-differentiable choose to return something useful rather than erroring.
    For a branch a function is not differentiable due to e.g. a branch, like `abs`, your rule can reasonably claim the derivative at that point is the value from either branch, *or* any value in-between.
    In particular for local optima (like in the case of `abs`) claiming the derivative is 0 is a good idea.
    Similarly, if derivative is from one side is not defined, or is not finite, return the derivative from the other side.
    Throwing an error, or returning `NaN` is generally the least useful option.

However, contrary to what calculus says most autodiff systems will return an answer for such functions.
For example for: `abs_left(x) = (x <= 0) ? -x : x`, AD will say the derivative at `x=0` is `-1`.
Alternatively for:  `abs_right(x) = (x < 0) ? -x : x`, AD will say the derivative at `x=0` is `1`.
Those two examples are weird since they are equal at all points, but AD claims different derivatives at `x=0`.
The way to fix autodiff systems being weird is to write custom rules.
So what rule should we write for this case?

The obvious answer, would be to write a rule that throws an error if input at a point where calculus says the derivative is not defined.
Another option is to return some error signally value like `NaN`.
Which you *can* do.
However, there is no where to go with an error, the user still wants a derivative; so this is not useful.

Let us explore what is useful:
## Case Studies

```@setup nondiff
using Plots
gr(framestyle=:origin, legend=false)
```
### Derivative is defined in usual sense
```@example nondiff
plot(x->x^3)
```
This is the standard case, one can return the derivative that is defined according to school room calculus.
Here we would reasonably say that at `x=0` the derivative is `3*0^2=0`. 



### Local Minima / Maxima

```@example nondiff
plot(abs)
```

`abs` is the classic example of a function where the derivative is not defined, as the limit from above is not equal to the limit from below.

$$\operatorname{abs}'(0) = \lim_{h \to 0^-} \dfrac{\operatorname{abs}(0)-\operatorname{abs}(0-h)}{0-h} = -1$$
$$\operatorname{abs}'(0) = \lim_{h \to 0^+} \dfrac{\operatorname{abs}(0)-\operatorname{abs}(0-h)}{0-h} = 1$$

Now, as discussed in the introduction, the AD system would on it's own choose either 1 or -1, depending on implementation.

We however have a potentially much nicer answer available to use: 0.

This has a number of advantages.
- It follows the rule that derivatives are zero at local minima (and maxima).
- If you leave a gradient descent optimizer running it will eventually actually converge absolutely to the point -- where as with it being 1 or -1 it would never outright converge it would always flee.

Further:
- It is a perfectly nice member of the [subderivative](https://en.wikipedia.org/wiki/Subderivative).
- It is the mean of the derivative on each side; which means that it will agree with central finite differencing at the point.
### Piecewise slope change
```@example nondiff
plot(x-> x < 0 ? x : 5x)
```

Here we have 3 main options, all are good.

We could say the derivative at 0 is:
 - 1: which agrees with backwards finite differencing
 - 5: which agrees with forwards finite differencing
 - 3: which is the mean of `[1, 5]`, and agrees with central finite differencing

All of these options are perfectly nice members of the [subderivative](https://en.wikipedia.org/wiki/Subderivative).
`3` is the arguably the nicest, but it is also the most expensive to compute.
In general all are acceptable.


### Derivative zero almost everywhere

```@example nondiff
plot(ceil)
```

Here it is most useful to say the derivative is zero everywhere.
The limits are zero from both sides.

The other option for `x->ceil(x)` would be to relax the problem into `x->x`, and thus  say it is 1 everywhere.
But that it too weird, if the user wanted a relaxation of the problem then they would provide one.
We can not be imposing that relaxation on to `ceil`, as it is not reasonable for everyone.

### Not defined on one-side
```@example nondiff
plot(x->exp(2log(x)))
plot!(; xlims=(-10,10), ylims=(-10,10)) #hide
```

We do not have to worry about what to return for the side where it is not defined.
As we will never be asked for the derivative at e.g. `x=-2.5` since the primal function errors.
But we do need to worry about at the boundary -- if that boundary point doesn't error.

Since we will never be asked about the left-hand side (as the primal errors), we can use just the right-hand side derivative.
In this case giving 0.0.

Also nice in this case is that it agrees with the symbolic simplification of `x->exp(2log(x))` into `x->x^2`.


### Derivative nonfinite and same on both sides

```@example nondiff
plot(cbrt)
```

Here we have no real choice but to say the derivative at `0` is `Inf`.
We could consider as an alternative saying some large but finite value.
However, if too large it will just overflow rapidly anyway; and if too small it will not dominate over finite terms.
It is not possible to find a given value that is always large enough.
Our alternatives would be to consider the derivative at `nextfloat(0.0)` or `prevfloat(0.0)`.
But this is more or less the same as choosing some large value -- in this case an extremely large value that will rapidly overflow.


### Derivative on-finite and different on both sides

```@example nondiff
plot(x-> sign(x) * cbrt(x))
```

In this example, the primal is defined and finite, so we would like a derivative to be defined.
We are back in the case of a local minimum like we were for `abs`.
We can make most of the same arguments as we made there to justify saying the derivative is zero.

## Conclusion

From the case studies a few general rules can be seen for how to choose a value that is _useful_.
These rough rules are:
 - Say the derivative is 0 at local optima.
 - If the derivative from one side is defined and the other isn't, say it is the derivative taken from the defined side.
 - If the derivative from one side is finite and the other isn't, say it is the derivative taken from the finite side.
 - When derivative from each side is not equal, strongly consider reporting the average.

Our goal as always, is to get a pragmatically useful result for everyone, which must by necessity also avoid a pathological result for anyone.
