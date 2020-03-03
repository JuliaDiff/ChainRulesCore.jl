# Debug Mode

ChainRules supports a [`debug_mode`](@ref) which you can use while writing new rules.
It provides better error messages.
If you are developing some new rules, and you get a weird error message,
it is worth enabling debug mode.

There is some overhead to having it enabled, so it is disabled by default.

To enable redefine the `debug_mode()` function to return `true`.
```julia
ChainRulesCore.debug_mode() = true
```
