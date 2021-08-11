# Debug Mode

ChainRulesCore supports a *debug mode* which you can use while writing new rules.
It provides better error messages.
If you are developing some new rules, and you get a weird error message,
it is worth enabling debug mode.

There is some overhead to having it enabled, so it is disabled by default.

To enable, redefine the [`ChainRulesCore.debug_mode`](@ref) function to return `true`.
```julia
ChainRulesCore.debug_mode() = true
```

## Features of Debug Mode:

 - If you add a `Tangent` to a primal value, and it was unable to construct a new primal values, then a better error message will be displayed detailing what overloads need to be written to fix this.
 - during [`add!!`](@ref), if an `InplaceThunk` is used, and it runs the code that is supposed to run in place, but the return result is not the input (with updated values), then an error is thrown. Rather than silently using what ever values were returned.
