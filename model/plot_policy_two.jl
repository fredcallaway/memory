include("plots_base.jl")
include("exp2_base.jl")

prm = load_fit("optimal", "results/sep11/exp1")
prm = (;prm..., switch_cost=prm.sample_cost)
m = exp2_mdp(prm)
B = BackwardsInduction(m; verbose=true)
pol = OptimalPolicy(B)

# %% --------

colors = [
    colorant"#D473A2",
    colorant"#3B77B3",
    colorant"#EAD56F",
]

evidences = collect(-m.threshold:B.dv:m.threshold)
to_idx_space(x) = cld(length(evidences), 2) + x / B.dv

function plot_policy(e2, t2; kws...)
    Q = B.Q[:, 1, :, e2, :, t2]
    V = B.V[1, :, e2, :, t2]
    delta = Q[1, :, :] - Q[2, :, :]
    X = fill(RGBA(0.,0.,0.,0.), size(delta))
    X[delta .≥ 0] .= colors[2]
    X[delta .< 0] .= colors[3]
    X[(V .≈ -m.miss_cost)] .= colors[1]

    plot(X;
        aspect_ratio=.7,
        size=(500, 300),
        yflip=false, 
        widen=false,
        xlab="", ylab="",
        yticks=[],
        xticks=[],
        kws...
    )
end

function plot_sim(beliefs, color=:black)
    x = Float64[]
    for b in beliefs
        b.focused != 1 && break
        push!(x, b.evidence[1])
    end
    x = min.(x, m.threshold + pol.B.dv)
    plot!(to_idx_space.(x[1:end]); color, lw=1.5)
end

function plot_predictions(ev2, t2; strong_only=false)
    t2 = 10
    b = Belief{2}(1, 1, [0., ev2], [0, t2])
    
    strengths = [
        (-0.02, -1.),
        (0.015, -1.)
    ]
    seeds = [3, 26]

    e2, t2 = belief2index(B, b)[[3, 5]]
    plot_policy(e2, t2, xlim=(1, size(B.V, 4) - t2))

    for (s, seed, color) in zip(strengths, seeds, RGB.([0.3, .6]))
        strong_only && s[1] < 0 && continue
        Random.seed!(seed)
        beliefs = simulate(pol; b=deepcopy(b), s, belief_log=BeliefLog2()).belief_log.beliefs
        plot_sim(beliefs, color)

        len = .5; stretch = .04
        s = s[1]
        x = √(len / (stretch^2 + s^2))
        plot!([1, x+1], to_idx_space.([0, x*s]); color, lw=1, ls=:dash, arrow=true)
    end
end

# %% --------


fig("exp2_policy_0", pdf=false) do
    plot_predictions(-.3, 30)
end

fig("exp2_policy_1", pdf=false) do
    plot_predictions(.1, 30, strong_only=true)
end
