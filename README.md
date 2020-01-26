<img src="https://rawcdn.githack.com/JuliaDiff/ChainRulesCore.jl/b0b8dbf26807f8f6bc1a3c073b6720b8d90a8cd4/docs/src/assets/logo.svg" width="256"/>

# ChainRulesCore

[![Travis](https://travis-ci.org/JuliaDiff/ChainRulesCore.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ChainRulesCore.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaDiff/ChainRulesCore.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaDiff/ChainRulesCore.jl?branch=master)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/C/ChainRulesCore.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

**Docs:**
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://JuliaDiff.github.io/ChainRulesCore.jl/dev)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDiff.github.io/ChainRulesCore.jl/stable)

The ChainRulesCore package provides a light-weight dependency for defining sensitivities for functions in your packages, without you needing to depend on ChainRules itself.

This will allow your package to be used with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl), which aims to provide a variety of common utilities that can be used by downstream automatic differentiation (AD) tools to define and execute forward-, reverse-, and mixed-mode primitives.

This package is a work in progress; the framework is essentially there, but there are a bunch of TODOs, virtually no tests, etc. PRs welcome! The API is mostly documented, which should help if you'd like to contribute.
