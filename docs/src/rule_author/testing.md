# Testing your rules

[ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl)
provides tools for writing tests based on [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).
Take a look at the documentation or the existing [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) tests to see how to write the tests.

!!! warning
    Don't use analytical derivations for derivatives in the tests.
    Those are what you use to define the rules, and so cannot be confidently used in the test.
    If you misread/misunderstood them, then your tests/implementation will have the same mistake.
    Use finite differencing methods instead, as they are based on the primal computation.
