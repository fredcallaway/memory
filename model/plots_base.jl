using Random

include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")
include("common.jl")

pyplot(dpi=300, size=(400,300), lw=2, grid=false, widen=true,
    background_color=:transparent, foreground_color=:black)