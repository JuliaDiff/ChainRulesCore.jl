<img src="https://rawcdn.githack.com/JuliaDiff/ChainRulesCore.jl/b0b8dbf26807f8f6bc1a3c073b6720b8d90a8cd4/docs/src/assets/logo.svg" width="256"/>

# ChainRulesCore

[![Build Status](https://github.com/JuliaDiff/ChainRulesCore.jl/workflows/CI/badge.svg)](https://github.com/JuliaDiff/ChainRulesCore.jl/actions?query=workflow:CI)
[![Coverage](https://codecov.io/gh/JuliaDiff/ChainRulesCore.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDiff/ChainRulesCore.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![DOI](https://zenodo.org/badge/199721843.svg)](https://zenodo.org/badge/latestdoi/199721843)

**Docs:**
[![](https://img.shields.io/badge/docs-main-blue.svg)](https://juliadiff.org/ChainRulesCore.jl/dev)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/ChainRulesCore.jl/stable)

The ChainRulesCore package provides a light-weight dependency for defining sensitivities for functions in your packages, without you needing to depend on ChainRules itself.

This will allow your package to be used with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl), which aims to provide a variety of common utilities that can be used by downstream automatic differentiation (AD) tools to define and execute forward-, reverse-, and mixed-mode primitives.

This package is a work in progress; PRs welcome!
