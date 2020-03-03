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
