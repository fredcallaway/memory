using Random

include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")

include("common.jl")

OUT_PATH = "/Users/fred/Papers/meta-memory/model-diagram"
if !isdir(OUT_PATH)
    @warn "$OUT_PATH does not existing, using figs/ by default"
    OUT_PATH = "figs"
end
mkpath(OUT_PATH)

pyplot(dpi=300, size=(400,300), lw=2, grid=false, widen=true,
    background_color=:transparent, foreground_color=:black)
fig(f, name; pdf=true, kws...) = figure(f, name; base=OUT_PATH, pdf, kws...)
