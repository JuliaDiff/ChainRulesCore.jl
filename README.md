# ChainRulesCore

[![Travis](https://travis-ci.org/JuliaDiff/ChainRulesCore.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ChainRulesCore.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaDiff/ChainRulesCore.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaDiff/ChainRulesCore.jl?branch=master)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaDiff.github.io/ChainRules.jl/latest)

The ChainRulesCore package provides a light-weight dependency for defining sensitivities for functions in your packages, without you needing to depend on ChainRules itself.

This will allow your package to be used with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl), which aims to provide a variety of common utilities that can be used by downstream automatic differentiation (AD) tools to define and execute forward-, reverse-, and mixed-mode primitives.

This package is a work in progress; the framework is essentially there, but there are a bunch of TODOs, virtually no tests, etc. PRs welcome! The API is mostly documented, which should help if you'd like to contribute.

The ChainRulesCore source code follows the [YASGuide](https://github.com/jrevels/YASGuide).
