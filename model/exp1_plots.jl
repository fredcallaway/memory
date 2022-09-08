using Random

include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")

include("common.jl")
include("exp1_base.jl")


pyplot(label="", dpi=300, size=(400,300), lw=2, grid=false, widen=true, 
    background_color=:transparent, foreground_color=:black)
fig(f, name; pdf=true, kws...) = figure(f, name; base="/Users/fred/Papers/meta-memory/model-diagram", pdf, kws...)

# prm = deserialize("tmp/exp1_opt_prm")
# m = exp1_mdp(prm)
# pol = OptimalPolicy(m)

# %% ==================== accumulation ====================

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

tracked = [1,]
tracked = [1, 14, 31, 60]

fig("accumulation1") do
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
    fig("b$i", widen=false, yaxis=true, size=(150,120), framestyle=:axes) do
        plot!(ylim=(0,2.5), xlim=(-2,2), xticks=-2:2)
        vline!([0], color=:black, lw=1)
        plot!(posterior(m, b)[1], color=colors[1], yticks=false, yaxis=false, xlabel="strength")
         # ylabel="Probability", 
    end
end

# %% ==================== policy heatmap ====================

# m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.05, 
#     threshold=8, noise=1.6, max_step=100, prior=Normal(0, 2)
# )
# pol = OptimalPolicy(m)
# B = pol.B

m = MetaMDP{1}(allow_stop=true, miss_cost=2, sample_cost=.01, 
    threshold=7, noise=.5, max_step=100, prior=Normal(0, 1)
)
B = BackwardsInduction(m; dv=.05)
pol = OptimalPolicy(B)

# %% --------

Random.seed!(5)
strengths = [-.5, 0, .2, 1]
sims = map(strengths) do s
    simulate(pol; s=(s,), belief_log=BeliefLog2()).belief_log.beliefs
end

evidences = collect(-m.threshold:B.dv:m.threshold)
to_idx_space(x) = cld(length(evidences), 2) + x / B.dv
idx = Int.(range(1, length(evidences), length=5))
colors = RGB.([0.3, .6, .8, 1])

function plot_policy()
    V = B.V[1, :, :]
    X = fill(colorant"#D473A2", size(V))
    X[V .> -m.miss_cost] .= colorant"#3B77B3"
    plot!(X,
        aspect_ratio=:.2,
        yflip=false,
        xlab="",
        ylab="",
        yticks=[],
        xticks=[],
    )
end

function plot_sim(i)
    sim = sims[i]; s = strengths[i]
    x = map(b-> b.evidence[1], sim)
    if sim[end].focused == -1
        x = x[1:end-1]
    end
    x = min.(x, m.threshold + pol.B.dv)
    plot!(to_idx_space.(x[1:end]), color=colors[i], lw=1.5)
    
    x = 15 / √(1 + s^2 * 15) # trying to get length about the same (too lazy to do math right)
    plot!([1, x+1], to_idx_space.([0, x] .* s), color=colors[i], lw=1, ls=:dash, arrow=true)
end

fig("exp1_policy_predictions", pdf=false, size=(500, 300), grid=false, widen=false) do
    plot_policy()
    for i in eachindex(sims)
        plot_sim(i)
    end
end

# %% --------
mkpath("figs/exp1_prediction_slides")
figure("exp1_prediction_slides/1", size=(600, 300), grid=false, widen=false) do
    plot_policy()
end

figure("exp1_prediction_slides/2", size=(600, 300), grid=false, widen=false) do
    plot_policy()
    plot_sim(3)
    plot_sim(4)
end

figure("exp1_prediction_slides/3", size=(600, 300), grid=false, widen=false) do
    plot_policy()
    plot_sim(1)
    plot_sim(2)
end
# %% --------
figure("exp1_prediction_slides/4", size=(600, 300), grid=false, widen=false) do
    # plot_policy()
    plot!(
        xlim=(0,100),
        xlab="time",
        ylab="recall progress",
        yticks=(idx, string.(evidences[idx]))
    )
    plot_sim(1)
    plot_sim(2)
    plot_sim(3)
    plot_sim(4)
end
