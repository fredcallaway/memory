using Random

include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")

include("common.jl")
include("exp1_base.jl")


pyplot(label="", dpi=300, size=(400,300), lw=2, grid=false, widen=true, 
    background_color=:transparent, foreground_color=:black)
fig(f, name; kws...) = figure(f, name; base="/Users/fred/Papers/meta-memory/diagram", pdf=true, kws...)

# prm = deserialize("tmp/exp1_opt_prm")
# m = exp1_mdp(prm)
# pol = OptimalPolicy(m)

# %% ==================== accumulation ====================

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.01, 
    threshold=8, noise=1, max_step=100, prior=Normal(0, 2)
)
pol = OptimalPolicy(m)

# %% --------

Random.seed!(9)

strengths = [-.3, 0., .5]
colors = ["#174675", "#4081C2","#51A6FB"]
fig("accumulation") do
    plot(size=(500, 230), xlim=(0,80), yaxis=(-8:4:8, (-12, 9)), xlab="Time", ylab="Evidence", widen=false)
    for (i, s) in enumerate(strengths)
        g = simulate(pol; s=(s,), belief_log=BeliefLog2()).belief_log.beliefs
        x = map(b-> b.evidence[1], g)
        x = min.(x, 8)
        plot!(x, color=colors[i])
    end
    hline!([8], color=:black)
end

# %% --------

Random.seed!(9)

fig("prediction", framestyle=:box) do
    plot!(size=(200, 200), xlim=(0,80), yaxis=(-8:4:8, (-12, 8)), xlab="Time", ylab="Evidence")
    for (i, s) in enumerate(strengths)
        g = simulate(pol; s=(s,), belief_log=BeliefLog2()).belief_log.beliefs
        if i == 2
            x = map(b-> b.evidence[1], g)
            x = min.(x, 8)
            plot!(x, color=colors[i])
            b = g[end]
            post = posterior(m, b)[1]
            ev = b.evidence[1]
            t = b.time[1] + 1.5
            tt = 0:.1:(m.max_step - t)

            σ = post.σ .* tt
            lo = ev .+ post.μ .* tt .+ σ
            hi = ev .+ post.μ .* tt .- σ

            # hi = min(threshold, hi)
            # if h < threshold
            # plot!(t .+ tt, ev .+ post.μ .* tt)

            c = colors[i]
            plot!(collect(t .+ tt), collect((ev .+ post.μ .* tt)); c, alpha=0.3, ls=:dot)
            plot!(t .+ tt, lo; c, fillrange=hi, fillalpha=0.15, alpha=0)

        end
    end
    hline!([8], color=:black)
    plot!(xlim=(40, 80), ylim=(-12, 2))
end

# %% ==================== strength distribution ====================

fig("strength_dist", widen=false, yaxis=true, size=(200,200), framestyle=:box) do
    # vline!([0], color=:black, lw=1,)
    plot!(ylim=(0,1), xlim=(-4,4))
    plot!(Normal(-1, 0.5), color="#4081C2", yticks=false, ylabel="Probability", xlabel="Strength")
end

# %% ==================== MDP state plots ====================

# m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.2, 
#     threshold=2, noise=.3, max_step=10, prior=Normal(0, .5)
# )
# pol = OptimalPolicy(m)

function plot_posterior!(post, ev, t, c; max_step, threshold)
    tt = 0:.1:(max_step - t)
    
    # var_sample = post.σ^2 + m.noise^2
    # σ = .√(var_sample .* tt)
    # lo = ev .+ post.μ .* tt .+ σ
    # hi = ev .+ post.μ .* tt .- σ

    σ = post.σ .* tt
    lo = ev .+ post.μ .* tt .+ σ
    hi = ev .+ post.μ .* tt .- σ

    # hi = min(threshold, hi)
    # if h < threshold
    # plot!(t .+ tt, ev .+ post.μ .* tt)

    plot!(collect(t .+ tt), collect((ev .+ post.μ .* tt)); c, alpha=0.3)
    plot!(t .+ tt, lo; c, fillrange=hi, fillalpha=0.15, alpha=0)
    scatter!([t], [ev]; c, markersize=5)
end

function plot_belief(b::Belief; threshold=m.threshold, max_step=m.max_step)
    plot(size=(120,120), dpi=500, xticks=false, yticks=false, 
        xlim=(-.1max_step,1.1max_step), ylim=(-1.1threshold,1.1threshold), 
        framestyle = :none)

    # hline!([0.001, threshold], line=(:black,), )
    # vline!([0.001, max_step], line=(:black,), )

    plot!([0, max_step, max_step], [-threshold, -threshold, threshold], line=(:black,))
    plot!([0, 0, max_step*1.005], [-threshold, threshold, threshold], line=(:black,))
    post = posterior(m, b)
    for i in eachindex(post)
        plot_posterior!(post[i], b.evidence[i], b.time[i], i; max_step, threshold)
    end
    # savefig("vec-figs/$name")
end

b = initial_belief(m)
fig("progress") do
    b.time[1] = 30
    b.evidence[1] = .3
    plot_belief(b)
end
# %% --------



Random.seed!(2)
function pick()
    while true
        g = simulate(pol; s=(0.,), belief_log=BeliefLog2()).belief_log.beliefs
        if length(g) > 5 && g[end].focused == 1
            return g
        end
    end
end



# g = pick()

mkpath("figs/beliefs")
for i in eachindex(g)
    plot_belief(g[i])
    savefig("figs/beliefs/$i")
end


# %% ==================== Policy heatmap ====================

# m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.015, threshold=40, noise=3, max_step=80)


m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.1, 
    threshold=8, noise=2, max_step=60, prior=Normal(0.5, 2)
)
# %% --------

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.06, 
    threshold=7, noise=1, max_step=75, prior=Normal(0, 1)
)

# m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.03, 
#     threshold=14, noise=2, max_step=150, prior=Normal(0, 1)
# )

# %% --------
B.V |> size
B = BackwardsInduction(m)

# %% --------
evidences = collect(-m.threshold:B.dv:m.threshold)
idx = Int.(range(1, length(evidences), length=5))

figure("exp1_policy") do
    heatmap(
        B.V[1, :, :],
        c=:RdBu_11, cbar=false,
        xlab="Time",
        ylab="Evidence",
        yticks=(idx, string.(evidences[idx]))
    )
end

# %% --------

figure("exp1_policy") do
    heatmap(
        B.V[1, :, :] .> -3,
        c=:RdBu_11, cbar=false,
        xlab="Time",
        ylab="Evidence",
        yticks=(idx, string.(evidences[idx]))
    )
end
