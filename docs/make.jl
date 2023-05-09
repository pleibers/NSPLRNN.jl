using NSPLRNN
using Documenter

DocMeta.setdocmeta!(NSPLRNN, :DocTestSetup, :(using NSPLRNN); recursive=true)

makedocs(;
    modules=[NSPLRNN],
    authors="Patrick Leibersperger <pleibersperger@posteo.de> and contributors",
    repo="https://github.com/pleibers/NSPLRNN.jl/blob/{commit}{path}#{line}",
    sitename="NSPLRNN.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pleibers.github.io/NSPLRNN.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pleibers/NSPLRNN.jl",
    devbranch="main",
)
