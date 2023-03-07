include("plots_base.jl")
include("exp1_base.jl")

x = load_fit("optimal", "results/sep11/exp1")
m = exp1_mdp(x)
B = BackwardsInduction(m)
pol = OptimalPolicy(B)

# %% --------

Random.seed!(5)
strengths = m.prior.σ .* [-.5, 0, .2, 1]
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
        aspect_ratio=.5,
        yflip=false,
        xlab="",
        ylab="",
        yticks=[],
        xticks=[],
    )
end

function plot_sim(i)
    sim = sims[i]; s = strengths[i]
    z = map(b-> b.evidence[1], sim)
    if sim[end].focused == -1
        z = z[1:end-1]
    end
    z = min.(z, m.threshold + pol.B.dv)
    plot!(to_idx_space.(z[1:end]), color=colors[i], lw=1.5)

    len = .5; stretch = .04
    x = √(len / (stretch^2 + s^2))
    plot!([1, x+1], to_idx_space.([0, x*s]), color=colors[i], lw=1, ls=:dash, arrow=true)
end

fig("exp1_policy_predictions", pdf=false, size=(600, 200), grid=false, widen=false, dpi=400) do
    plot_policy()
    for i in eachindex(sims)
        plot_sim(i)
    end
end
