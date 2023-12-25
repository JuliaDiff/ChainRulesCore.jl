# Team

The team behind the ChainRules project is semi-fluid and fuzzy around the edges.
It is not a concrete formally defined group from one company or organization.

Here we list people who have merge rights, and are regular maintainers of the project and its constituent parts.

````@eval
team = [
    (github="oxinabox", org="Invenia Labs", subtext="Project Lead"),
    (github="mzgubic", org="Invenia Labs", subtext=""),
    (github="mcabbott", org="", subtext=""),
    (github="willtebbutt", org="Invenia Labs", subtext=""),
    (github="sethaxen", org="Universität Tübingen", subtext=""),
    (github="ararslan", org="Beacon Biosignals", subtext="(retired)"),
    (github="jrevels", org="Beacon Biosignals", subtext="Original Creator (retired)"),
]

using Markdown
using Cascadia
using Gumbo
using Downloads: download



read_url(url) = parsehtml(String(take!(download(url, IOBuffer()))))

function get_avatar_url(doc)
    eles = eachmatch(sel"img.avatar-user", doc.root)
    return getattr(first(eles), "src")
end

function get_text(doc, selector)
        eles = eachmatch(selector, doc.root)
        return text(only(eles))
end

OUT = IOBuffer()
for person in team
    profile = joinpath("https://github.com", person.github)
    doc = read_url(profile)
    full_name = get_text(doc, sel"span.vcard-fullname")
    avatar = get_avatar_url(doc)
    org_text = isempty(person.org) ? "" : "_$(person.org)_"

    print(OUT, " - ")
    print(OUT, "[![$(person.github)]($(avatar))]($profile)")
    print(OUT, " **$(full_name)** $(org_text) $(person.subtext)")
    println(OUT)
end

Markdown.parse(String(take!(OUT)))
````
