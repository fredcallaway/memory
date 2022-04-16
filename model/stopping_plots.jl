include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")


# %% ==================== Accumulation plot ====================

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.05, 
    threshold=8, noise=2, max_step=60, prior=Normal(0, 2)
)

using Random
Random.seed!(9)
strengths = [-1, 0, 2.]
figure(xlim=(0,40), yaxis=(-8:4:8, (-8, 8)), widen=true, xlab="Time", ylab="Evidence") do
    for (i, s) in enumerate(strengths)
        for j in 1:3
            g = simulate(pol; s=(s,), belief_log=BeliefLog2()).belief_log.beliefs
            x = map(b-> b.evidence[1], g)
            plot!(x, color=i, lw=1)
        end
    end
    hline!([8], color=:black)
end


# %% ==================== MDP state plots ====================

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.2, 
    threshold=2, noise=.3, max_step=10, prior=Normal(0, .5)
)
pol = OptimalPolicy(m)

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

Random.seed!(2)
function pick()
    while true
        g = simulate(pol; s=(0.,), belief_log=BeliefLog2()).belief_log.beliefs
        if length(g) > 5 && g[end].focused == 1
            return g
        end
    end
end

g = pick()

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

m = MetaMDP{1}(allow_stop=true, miss_cost=2, sample_cost=.06, 
    threshold=7, noise=1, max_step=100, prior=Normal(0, 1)
)

# m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.03, 
#     threshold=14, noise=2, max_step=150, prior=Normal(0, 1)
# )


B = BackwardsInduction(m; dv=.05)
pol = OptimalPolicy(B)

evidences = collect(-m.threshold:B.dv:m.threshold)
idx = Int.(range(1, length(evidences), length=5))

figure("exp1_policy", size=(600, 300), grid=false, widen=false) do
    heatmap!(
        B.V[1, :, :] .> -m.miss_cost, 
        c=["#D473A2","#3B77B3"], cbar=false,
        xlab="time",
        ylab="recall progress",
        yticks=(idx, string.(evidences[idx]))
    )
end

# %% --------
evidences = collect(-m.threshold:B.dv:m.threshold)

times = 0:200


bs = map(product(evidences, times)) do (e, t)
    Belief{1}(t, 1, [e], [t])
end

V = map(bs) do b
    value(B, b) > -3
end

μ = map(bs) do b
    mean(posterior(m, b)[1])
end

σ = map(bs) do b
    std(posterior(m, b)[1])
end

df = DataFrame((;μ, σ, V))

# %% --------
using Optim

predict_v(μ, σ, θ) = cdf.(Normal.(μ, σ), θ[1]) .< θ[2]
res = optimize([0.1, 0.1]) do θ
    mean(predict_v(μ, σ, θ) .!= V)
end
res.minimizer, res.minimum

# stop when  P(strength < θ1) > θ2

# %% --------
predict_v(μ, σ, θ) = μ .+ θ[1] .* σ .> θ[2]

res2 = optimize(0, 1) do θ1
    mean(predict_v(μ, σ, [θ1, 0.]) .!= V)
end
res2.minimizer, res2.minimum


# %% --------

quantize(x, q) = @chain x begin
    cld(q)
    q * _
    round(;digits=6)
    _ === -0. ? 0. : _
end

X = @chain df begin
    @rtransform begin
        :μ = quantize(:μ, .05)
        :λ = :σ ^- 2
    end
    @rsubset abs(:μ) ≤ 1
    @bywrap [:μ, :λ] mean(:V)
end

figure() do
    heatmap(X)
end


# %% --------

P = map(grid(μ=-1:.001:1, λ=1:201)) do (;μ, λ)
    σ = λ^-0.5
    predict_v(μ, σ, [.005, .7])
end
res.minimizer

figure() do
    heatmap(P)
end


