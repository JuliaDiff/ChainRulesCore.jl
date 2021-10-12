using ChainRulesCore
using Documenter
using DocThemeIndigo
using Markdown

DocMeta.setdocmeta!(
    ChainRulesCore,
    :DocTestSetup,
    quote
        using Random
        Random.seed!(0)  # frule doctest shows output

        using ChainRulesCore, LinearAlgebra
        # These rules are all actually defined in ChainRules.jl, but we redefine them here to
        # avoid the dependency.
        @scalar_rule(sin(x), cos(x))  # frule and rrule doctest
        @scalar_rule(sincos(x), @setup((sinx, cosx) = Ω), cosx, -sinx)  # frule doctest
        @scalar_rule(hypot(x::Real, y::Real), (x / Ω, y / Ω))  # rrule doctest
    end,
)

indigo = DocThemeIndigo.install(ChainRulesCore)

makedocs(;
    modules=[ChainRulesCore],
    format=Documenter.HTML(;
        prettyurls=false,
        assets=[indigo],
        mathengine=MathJax3(
            Dict(
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    # TODO: remove when using physics package
                    "macros" => Dict(
                        "ip" => ["{\\left\\langle #1, #2 \\right\\rangle}", 2],
                        "Re" => "{\\operatorname{Re}}",
                        "Im" => "{\\operatorname{Im}}",
                        "tr" => "{\\operatorname{tr}}",
                    ),
                ),
            ),
        ),
    ),
    sitename="ChainRules",
    authors="Jarrett Revels and other contributors",
    pages=[
        "Introduction" => "index.md",
        "FAQ" => "FAQ.md",
        "Rule configurations and calling back into AD" => "config.md",
        "Opting out of rules" => "opting_out_of_rules.md",
        "Writing Good Rules" => "writing_good_rules.md",
        "Complex Numbers" => "complex.md",
        "Deriving Array Rules" => "arrays.md",
        "Debug Mode" => "debug_mode.md",
        "Gradient Accumulation" => "gradient_accumulation.md",
        "Usage in AD" => "use_in_ad_system.md",
        "Converting ZygoteRules" => "converting_zygoterules.md",
        "Tips for making packages work with AD" => "tips_for_packages.md",
        "Design" => [
            "Changing the Primal" => "design/changing_the_primal.md",
            "Many Differential Types" => "design/many_differentials.md",
        ],
        "API" => "api.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/JuliaDiff/ChainRulesCore.jl.git", devbranch="main", push_preview=true
)
