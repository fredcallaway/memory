using Random

include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")

include("common.jl")
include("exp2_base.jl")

pyplot(label="", dpi=300, size=(400,300), lw=2, grid=false, widen=true, 
    background_color=:white, foreground_color=:black)

# %% --------

m = MetaMDP{2}(allow_stop=true, miss_cost=2, sample_cost=.01, switch_cost=.01,
    threshold=7, noise=.5, max_step=100, prior=Normal(0, 1)
)

B = BackwardsInduction(m; dv=0.2, verbose=true)

# %% ==================== accumulation ====================

Random.seed!(15)
pol = OptimalPolicy(B)
# colors = RGB.([.2, .6])
figure(size=(700, 300), widen=false) do
    plot!(
        xlabel="time",
        ylabel="recall progress",
        ylim=(-7.3, 7.3),
        xlim=(0,61),
    )
    θs = (0.25, 0.45)
    sim = simulate(pol; s=θs, belief_log=BeliefLog2())

    beliefs = sim.belief_log.beliefs
    
    for i in 1:2
        plot!(min.(m.threshold, map(b-> b.evidence[i], beliefs)))
    end
    hline!([m.threshold], color=:black)

    cs = getfield.(beliefs, :focused)
    switches = findall(diff(cs) .!= 0)

    vline!(findall(diff(cs) .!= 0), color=:gray, lw=0.5)
end


# %% ==================== policy heatmap ====================

# m = MetaMDP{2}(sample_cost=.1, threshold=4, noise=2, switch_cost=.1,
#         max_step=30, prior=Normal(0.5, 2)
# )

# m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.1, threshold=4, noise=2, switch_cost=0,
#         max_step=30, prior=Normal(0.5, 2)
# )

# m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.03, switch_cost=.03,
#     threshold=14, noise=2, max_step=100, prior=Normal(1, 1)
# )

m = MetaMDP{2}(allow_stop=true, miss_cost=2, sample_cost=.01, switch_cost=.01,
    threshold=7, noise=.5, max_step=100, prior=Normal(0, 1)
)

B = BackwardsInduction(m; dv=0.2, verbose=true)

# %% --------

colors = [
    colorant"#D473A2",
    colorant"#3B77B3",
    colorant"#EAD56F",
]

evidences = collect(-m.threshold:B.dv:m.threshold)
to_idx_space(x) = cld(length(evidences), 2) + x / B.dv
# idx = Int.(range(1, length(evidences), length=5))

function plot_policy(e2, t2; kws...)
    Q = B.Q[:, 1, :, e2, :, t2]
    V = B.V[1, :, e2, :, t2]
    delta = Q[1, :, :] - Q[2, :, :]
    X = fill(RGBA(0.,0.,0.,0.), size(delta))
    X[delta .≥ 0] .= colors[2]
    X[delta .< 0] .= colors[3]
    X[(V .≈ -m.miss_cost)] .= colors[1]
    plot(X;
        yflip=false, 
        widen=false,
        xlab=" ", ylab=" ",
        yticks=[],
        xticks=[],
        kws...
    )
end

function get_evidence(m, μ, t)
    λ_obs = m.noise ^ -2
    λ_prior = m.prior.σ^-2
    μ_prior = m.prior.μ
    λ = λ_obs * t + λ_prior
    (μ * λ - μ_prior * λ_prior) / λ_obs
end

function policy_fig(ev2, t2)
    b = Belief{2}(1, 1, [0., ev2], [0, t2])
    e2, t2 = belief2index(B, b)[[3, 5]]
    figure("exp2_policy_$(ev2)_$t2") do
        plot_policy(e2, t2,
            size=(300,300),
            xlab="time on current memory", 
            ylab="recall progress",
            xlim=(1, 91),
        )

        μ2 = posterior(m, ev2, t2).μ
        x = 0:(100-t2)
        y = to_idx_space.(get_evidence.([m], μ2, x))
        plot!(x, y, color=:white, lw=2, ls=:dash, ylim=to_idx_space.([-m.threshold, m.threshold]))
    end
end

# %% --------

policy_fig(-1, 10)
policy_fig(1, 10)
policy_fig(0, 10)


# %% ==================== policy (e1 x e2) ====================


function plot_policy(t1, t2; kws...)
    Q = B.Q[:, 1, :, :, t1, t2]
    V = B.V[1, :, :, t1, t2]
    delta = Q[1, :, :] - Q[2, :, :]
    X = fill(RGBA(0.,0.,0.,0.), size(delta))
    X[delta .≥ 0] .= colors[2]
    X[delta .< 0] .= colors[3]
    X[(V .≈ -m.miss_cost)] .= colors[1]
    plot(X;
        yflip=false, 
        widen=false,
        # ylab="progress on current",
        # xlab="progress on other",
        # yticks=(idx, string.(evidences[idx])),
        # xticks=(idx, string.(evidences[idx])),
        xlab=" ", ylab=" ",
        yticks=[],
        xticks=[],
        kws...
    )

end
figure("exp2_policy_25") do
    plot_policy(25, 25)
        size=(300,300),
        ylab="recall progress on current target", 
        xlab="recall progress on other target")
end
# %% --------
figure("exp2_policy_multi") do
    pp = [plot_policy(t1, t2) for t2 in [5, 25, 50], t1 in [50, 25, 5]]
    plot!(pp[4], ylab="time on current target")
    plot!(pp[8], xlab="time on other target")
    plot!(pp...;
        layout=(3, 3), size=(400, 400)
    )
end
# %% --------
function demo(x, y)
    plot(xticks=[], yticks=[])
    annotate!(.5, .5, text("$x, $y", 20))
end

figure() do
    plots = [demo(t1, t2) for t1 in [5, 25, 50], t2 in [50, 25, 5]]
    plot(plots...)

end

# %% --------
figure() do
    plot!(
        plot_policy(10, 10, xlab=""),
        plot_policy(20, 10, ylab=""),
        plot_policy(30, 10, xlab="", ylab=""),
        layout=(1,3), size=(750, 250)
    )
end

# %% ==================== old ====================
# %% --------



# %% --------
pol = OptimalPolicy(B)
monte_carlo() do
    sim = simulate(pol, fix_log=FullFixLog())
    [sim.b.focused != -1, sum(sim.fix_log.fixations), length(sim.fix_log.fixations)]
end

# %% --------
sims = map(1:10000) do i
    simulate(pol, fix_log=FullFixLog())
end;

# %% --------

nfix = map(sims) do sim
    length(sim.fix_log.fixations)
end

counts(nfix) ./ length(sims)




# %% --------
m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.03, switch_cost=.03,
    threshold=14, noise=2, max_step=60, prior=Normal(1, 1)
)

@time B = BackwardsInduction(m; dv=0.2, compute=false)

@time compute_value_functions!(B);
# %% --------



# %% --------
Q = B.Q[:, 1, :, 41, :, 1]
# X = Q[1, :, :] - Q[2, :, :]
# rng = maximum(filter(!isnan, (abs.(X))))
# figure() do
#     heatmap(X , c=:RdBu_11, cbar=false, clims=(-rng, rng))
# end

figure() do
    X = (Q[1, :, :] .> Q[2, :, :]) .* 2 .- 1
    X .*= .!((Q[1, :, :] .≈ Q[2, :, :]) .| isnan.(Q[1, :, :]))
    heatmap(X, c=:RdBu_11, cbar=false,
        xaxis="Time on Cue",
        yaxis="Accumulated Evidence"
    )
end


# %% ==================== Vector beliefs ====================
include("mdp.jl")
include("utils.jl")
include("figure.jl")
# %% --------

function plot_vec!(h, t, c; max_step, threshold)
    total = t+h
    scatter!([total], [h]; c, markersize=5)
    lo, hi = (1+max_step-total) .* quantile(Beta(1+h,1+t), [0.1, 0.9])
    # hi = min(threshold, hi)
    if h < threshold
        plot!([total, 1+max_step], [h, h+lo]; c, fillrange=[h, h+hi], fillalpha=0.3, alpha=0)
    end
end

function plot_belief(b::Belief; threshold=10, max_step=20)
    plot(size=(200,200), dpi=500, xticks=false, yticks=false, 
        xlim=(-.05max_step,1.05max_step), ylim=(-.05threshold,1.05threshold), 
        framestyle = :none)

    # hline!([0.001, threshold], line=(:black,), )
    # vline!([0.001, max_step], line=(:black,), )

    plot!([0, max_step, max_step], [0, 0, threshold], line=(:black,))
    plot!([0, 0, max_step*1.005], [0, threshold, threshold], line=(:black,))
    for i in eachindex(b.heads)
        plot_vec!(b.heads[i], b.tails[i], i; max_step, threshold)
    end
    # savefig("vec-figs/$name")
end


to_plot = [
    (0, 0)
    (1, 0)
    (4, 3)
    (10,7)
]
mkpath("figs/1belief")
for (h, t) in to_plot
    b = Belief{1}(0, 1, [h], [t])
    plot_belief(b)
    savefig("figs/1belief/$h-$t.png")
end


# %% --------
function plot_belief(name, h1, t1, h2, t2)
    @assert h1 ≤ t1 && h2 ≤ t2
    plot(size=(200,200), dpi=500, xticks=false, yticks=false, xlim=(-0.4,7), ylim=(-0.4,7), 
        aspect_ratio=:equal, framestyle = :zerolines)
    plot_vec!(h1, t1, 1)
    plot_vec!(h2, t2, 2)
    savefig("vec-figs/$name")
end


# h = 1; t = 1
# d =
mkpath("vec-figs")

plot_belief("one", 0,0,0,0)
plot_belief("two", 1, 1, 0, 0)
plot_belief("three", 1, 2, 6, 6)

# %% --------


function plot_vec!(h, t, c)
    scatter!([t], [h]; c, markersize=6)
    lo, hi = (60-t) .* quantile(Beta(1+h,1+t-h), [0.1, 0.9])
    @show lo hi
    plot!([t, 60], [h, h+lo]; c, fillrange=[h, h+hi], fillalpha=0.3, alpha=0)
    hline!([20], line=(:black, :dash))
end

function plot_belief(name, h1, t1, h2, t2)
    @assert h1 ≤ t1 && h2 ≤ t2
    plot(size=(400,250), dpi=500, xlim=(-0.4,60), ylim=(-0.4,20),
          xlabel="Time Fixated", ylabel="Accumulated Evidence")
    plot_vec!(h1, t1, 1)
    plot_vec!(h2, t2, 2)
    savefig("vec-figs/$name")
end

plot_belief("four", 1, 4, 2, 8)



# %% --------
mkpath("beta-figs")

function plot_beta!(h, t, c)
    d = Beta(1+h,1+t-h)
    x = 0:.001:1
    plot!(x, pdf.(d, x))
end

function plot_belief(name, h1, t1, h2, t2)
    @assert h1 ≤ t1 && h2 ≤ t2
    plot(size=(200,200), dpi=500, xticks=false, yticks=false, xlim=(0,1), ylim=(0,3))
    plot_beta!(h1, t1, 1)
    plot_beta!(h2, t2, 2)
    savefig("beta-figs/$name")
end

plot_belief("one", 0,0,0,0)
plot_belief("two", 1, 1, 0, 0)
plot_belief("three", 1, 2, 6, 6)

plot_belief("four", 1, 4, 2, 6)


