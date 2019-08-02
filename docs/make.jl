using ChainRulesCore
using Documenter

makedocs(modules=[ChainRulesCore],
         sitename="ChainRulesCore",
         authors="Jarrett Revels and other contributors",
         pages=["Introduction" => "index.md",
                "Getting Started" => "getting_started.md",
                "ChainRulesCore API Documentation" => "api.md"])

deploydocs(repo="github.com/JuliaDiff/ChainRulesCore.jl.git")
