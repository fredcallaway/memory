using Random

include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")

include("common.jl")
include("exp2_base.jl")

pyplot(label="", dpi=300, size=(400,300), lw=2, grid=false, widen=true,
    background_color=:white, foreground_color=:black)

# %% ==================== simulate ====================

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.02,
    threshold=8, noise=1.6, max_step=100, prior=Normal(0, 2)
)
pol = OptimalPolicy(m)

Random.seed!(7)

strengths = [0, 1.7]
colors = ["#174675","#51A6FB"]
sims = map(strengths) do s
    simulate(pol; s=(s,), belief_log=BeliefLog2()).belief_log.beliefs
end

tracked = [1, 14, 31, 60]

# %% ==================== accumulation ====================


figure("accumulation1") do
    plot(size=(700, 230), xlim=(0,62), yaxis=(-8:4:8, (-11, 9)), xlab=" ", ylab=" ", widen=false)
    for (i, sim) in enumerate(sims)

        x = map(b-> b.evidence[1], sim)
        x = min.(x, m.threshold)
        plot!(x, color=colors[i])
    end

    x = map(b-> b.evidence[1], sims[1])
    scatter!(tracked, x[tracked], markeralpha=0, markerstrokealpha=1, markersize=10)
    hline!([8], color=:black)
end


# %% ==================== belief states ====================


for (i, b) in enumerate(sims[1][tracked])
    figure("b$i", widen=false, yaxis=true, size=(150,120), framestyle=:axes) do
        plot!(ylim=(0,2.5), xlim=(-2,2), xticks=-2:2)
        vline!([0], color=:black, lw=1)
        plot!(posterior(m, b)[1], color=colors[1], yticks=false, yaxis=false, xlabel="strength")
         # ylabel="Probability",
    end
end