
# %% ==================== Policy Heatmap ====================

# m = MetaMDP{2}(sample_cost=.1, threshold=4, noise=2, switch_cost=.1,
#         max_step=30, prior=Normal(0.5, 2)
# )

# m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.1, threshold=4, noise=2, switch_cost=0,
#         max_step=30, prior=Normal(0.5, 2)
# )

m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.03, switch_cost=.03,
    threshold=14, noise=2, max_step=60, prior=Normal(1, 1)
)

B = BackwardsInduction(m; dv=0.2)

countmap([simulate(OptimalPolicy(B)).b.focused for i in 1:1000])

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


