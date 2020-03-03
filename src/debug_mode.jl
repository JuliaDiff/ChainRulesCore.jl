"""
    debug_mode() -> Bool

Determines if ChainRulesCore is in `debug_mode`.
Defaults to `false`, but if the user redefines it to return `true` then extra
information will be shown when errors occur.

Enable via:
```
ChainRulesCore.debug_mode() = true
```
"""
debug_mode() = false
