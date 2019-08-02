using AbstractChainRules
using Documenter

makedocs(modules=[AbstractChainRules],
         sitename="AbstractChainRules",
         authors="Jarrett Revels and other contributors",
         pages=["Introduction" => "index.md",
                "Getting Started" => "getting_started.md",
                "AbstractChainRules API Documentation" => "api.md"])

deploydocs(repo="github.com/JuliaDiff/AbstractChainRules.jl.git")
