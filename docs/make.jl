using ChainRulesCore
using Documenter

@show ENV

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

makedocs(
    modules=[ChainRulesCore],
    format=Documenter.HTML(prettyurls=false, assets=["assets/chainrules.css"]),
    sitename="ChainRules",
    authors="Jarrett Revels and other contributors",
    pages=[
        "Introduction" => "index.md",
        "FAQ" => "FAQ.md",
        "Writing Good Rules" => "writing_good_rules.md",
        "Debug Mode" => "debug_mode.md",
        "Design" => [
            "Many Differential Types" => "design/many_differentials.md",
        ],
        "API" => "api.md",
    ],
    strict=true,
    checkdocs=:exports,
)

const repo = "github.com/JuliaDiff/ChainRulesCore.jl.git"
const PR = get(ENV, "TRAVIS_PULL_REQUEST", "false")
if PR == "false"
    # Normal case, only deploy docs if merging to master or release tagged
    deploydocs(repo=repo)
else
    @info "Deploying review docs for PR #$PR"
    # TODO: remove most of this once https://github.com/JuliaDocs/Documenter.jl/issues/1131 is resolved

    # Overwrite Documenter's function for generating the versions.js file
    foreach(Base.delete_method, methods(Documenter.Writers.HTMLWriter.generate_version_file))
    Documenter.Writers.HTMLWriter.generate_version_file(_, _) = nothing
    # Overwrite necessary environment variables to trick Documenter to deploy
    ENV["TRAVIS_PULL_REQUEST"] = "false"
    ENV["TRAVIS_BRANCH"] = "master"

    deploydocs(
        devurl="preview-PR$(PR)",
        repo=repo,
    )
end
