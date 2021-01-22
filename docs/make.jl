using ChainRulesCore
using Documenter
using DocumenterTools: Themes
using Markdown

DocMeta.setdocmeta!(
    ChainRulesCore,
    :DocTestSetup,
    quote
        using Random
        Random.seed!(0)  # frule doctest shows output

        using ChainRulesCore
        # These rules are all actually defined in ChainRules.jl, but we redefine them here to
        # avoid the dependency.
        @scalar_rule(sin(x), cos(x))  # frule and rrule doctest
        @scalar_rule(sincos(x), @setup((sinx, cosx) = Ω), cosx, -sinx)  # frule doctest
        @scalar_rule(hypot(x::Real, y::Real), (x / Ω, y / Ω))  # rrule doctest
    end
)

Themes.compile(joinpath(@__DIR__, "src/assets/chainrules.scss"))

makedocs(
    modules=[ChainRulesCore],
    format=Documenter.HTML(
        prettyurls=false,
        assets=["assets/chainrules.css"],
        mathengine=MathJax(),
    ),
    sitename="ChainRules",
    authors="Jarrett Revels and other contributors",
    pages=[
        "Introduction" => "index.md",
        "FAQ" => "FAQ.md",
        "Writing Good Rules" => "writing_good_rules.md",
        "Complex Numbers" => "complex.md",
        "Deriving Array Rules" => "arrays.md",
        "Debug Mode" => "debug_mode.md",
        "Gradient Accumulation" => "gradient_accumulation.md",
        "Usage in AD" => [
            "Overview" => "autodiff/overview.md",
            "Operator Overloading" => "autodiff/operator_overloading.md",
        ],
        "Design" => [
            "Changing the Primal" => "design/changing_the_primal.md",
            "Many Differential Types" => "design/many_differentials.md",
        ],
        "API" => "api.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(
    repo = "github.com/JuliaDiff/ChainRulesCore.jl.git",
    push_preview=true,
)
