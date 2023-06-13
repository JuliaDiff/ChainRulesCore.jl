# Videos

For people who learn better by video we have a number of videos of talks we have given about the ChainRules project.
Note however, that the videos are frozen in time reflecting the state of the packages at the time they were recorded.
This documentation is the continuously updated canonical source.
However, we have tried to note below each video notes on its correctness.

The talks that follow are in reverse chronological order (i.e. most recent video is first).

### EuroAD 2021: ChainRules.jl: AD system agnostic rules for JuliaLang
Presented by Frames White.
[Slides](https://www.slideshare.net/LyndonWhite2/euroad-2021-chainrulesjl)

This is the talk to watch if you want to understand why the ChainRules project exists, what its challenges are, and how those have been overcome.
It is intended less for users of the package, and more for people working in the field of AD more generally.
It does also serve as a nice motivation for those first coming across the package as well though.

```@raw html
<div class="video-container">
<iframe src="https://www.youtube.com/embed/B3bC49OmTdk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

Abstract:
> The ChainRules project is a suite of JuliaLang packages that define custom primitives (i.e. rules) for doing AD in JuliaLang.
> Importantly it is AD system agnostic.
> It has proved successful in this goal.
> At present it works with about half a dozen different JuliaLang AD systems.
> It has been a long journey, but as of August 2021, the core packages have now hit version 1.0.
>
> This talk will go through why this is useful, the particular objectives the project had, and the challenges that had to be solved.
> This talk is not intended as an educational guide for users (For that see our 2021 JuliaCon talk: > Everything you need to know about ChainRules 1.0 (https://live.juliacon.org/talk/LWVB39)).
> Rather this talk is to share the insights we have had, and likely (inadvertently) the mistakes we have made, with the wider autodiff community.
> We believe these insights can be informative and useful to efforts in other languages and ecosystems.


### JuliaCon 2021: Everything you need to know about ChainRules 1.0
Presented by Miha Zgubič.
[Slides](https://github.com/mzgubic/ChainRulesTalk/blob/master/ChainRules.pdf)

If you are just wanting to watch a video to learn all about ChainRules and how to use it, watch this one.

!!! note "Slide on opting out is incorrect"
    Slide 42 is incorrect (`@no_rrule sum_array(A::Diagonal)`), in the ChainRulesCore 1.0 release the following syntax is used: `@opt_out rrule(::typeof(sum_array), A::Diagonal)`. This syntax allows us to include rule config information.

```@raw html
<div class="video-container">
<iframe src="https://www.youtube.com/embed/a8ol-1l84gc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

Abstract:
> ChainRules is an automatic differentiation (AD)-independent ecosystem for forward-, reverse-, and mixed-mode primitives. It comprises ChainRules.jl, a collection of primitives for Julia Base, ChainRulesCore.jl, the utilities for defining custom primitives, and ChainRulesTestUtils.jl, the utilities to test primitives using finite differences. This talk provides brief updates on the ecosystem since last year and focuses on when and how to write and test custom primitives.


### JuliaCon 2020: ChainRules.jl
Presented by Frames White.
[Slides](https://raw.githack.com/oxinabox/ChainRulesJuliaCon2020/main/out/build/index.html)

This talk is primarily of historical interest.
This was the first public presentation of ChainRules.
Though the project was a few years old by this stage.
A lot of things are still the same; conceptually, but a lot has changed.
Most people shouldn't watch this talk now.

!!! warning "Outdated Terminology"
    A lot of terminology has changed since this presentation.
     - `DoesNotExist` → `NoTangent`
     - `Zero` →  `ZeroTangent`
     - `Composite{P}` → `Tangent{T}`
    The talk also says differential in a lot of places where we now would say tangent.

```@raw html
<div class="video-container">
<iframe src="https://www.youtube.com/embed/B4NfkkkJ7rs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

Abstract:
> The ChainRules project allows package authors to write rules for custom sensitivities (sometimes called custom adjoints) in a way that is not dependent on any particular autodiff (AD) package.
> It allows authors of AD packages to access a wealth of prewritten custom sensitivities, saving them the effort of writing them all out themselves.
> ChainRules is the successor to DiffRules.jl and is the native rule system currently used by ForwardDiff2, Zygote and soon ReverseDiff
