ENV["GKSwstype"] = "100"  # make Plots/GR work on headless machine

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
        "How to use ChainRules as a rule author" => [
            "Introduction" => "rule_author/intro.md",
            "Pedagogical example" => "rule_author/example.md",
            "Tangent types" => "rule_author/tangents.md",
            "Which functions need rules?" => "rule_author/which_functions_need_rules.md",
            "Rule definition tools" => "rule_author/rule_definition_tools.md",
            "Writing good rules" => "rule_author/writing_good_rules.md",
            "Testing your rules" => "rule_author/testing.md",
            "Superpowers" => [
                "`ProjectTo`" => "rule_author/superpowers/projectto.md",
                "`@opt_out`" => "rule_author/superpowers/opt_out.md",
                "`RuleConfig`" => "rule_author/superpowers/ruleconfig.md",
                "Gradient accumulation" => "rule_author/superpowers/gradient_accumulation.md",
                "Mutation Support (experimental)" => "rule_author/superpowers/mutation_support.md",
            ],
            "Converting ZygoteRules.@adjoint to rrules" => "rule_author/converting_zygoterules.md",
            "Tips for making your package work with AD" => "rule_author/tips_for_packages.md",
            "Debug mode" => "rule_author/debug_mode.md",
        ],
        "How to support ChainRules rules as an AD package author" => [
            "Usage in AD" => "ad_author/use_in_ad_system.md",
            "Support calling back into ADs" => "ad_author/call_back_into_ad.md",
            "Support opting out of rules" => "ad_author/opt_out.md",
        ],
        "The maths" => [
            "The propagators: pushforward and pullback" => "maths/propagators.md",
            "Non-differentiable Points" => "maths/nondiff_points.md",
            "Complex numbers" => "maths/complex.md",
            "Deriving array rules" => "maths/arrays.md",
        ],
        "Design" => [
            "Changing the Primal" => "design/changing_the_primal.md",
            "Many Tangent Types" => "design/many_tangents.md",
        ],
        "Videos" => "videos.md",
        "FAQ" => "FAQ.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/JuliaDiff/ChainRulesCore.jl.git", devbranch="main", push_preview=true
)
