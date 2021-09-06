include("binomial_accumulator.jl")
include("figure.jl")
using StatsPlots

figure() do
    θs = (0.2, 0.5, 0.8)
    for (i, θ) in enumerate(θs)
        for j in 1:3
            m = MetaMDP{1}(step_size=1, threshold=50)
            pol = RandomPolicy(m)
            bs = simulate(pol; s = (θ,), save_beliefs=true).bs
            lab = j == 1 ? "θ = $θ" : ""
            plot!([b.counts[1][1] for b in bs]; c=i, alpha=1, lw=20, lab)
        end
    end
    hline!([50], color=:black, lab="Threshold")
    plot!(xlab="Time", ylab="Evidence", legend=false)
end

# %% --------
m = MetaMDP{2}(step_size=2, threshold=19, switch_cost=4, max_step=100)
pol = OptimalPolicy(m)
# %% --------

figure() do
    θs = (0.2, 0.3)
    sim = simulate(pol; s=θs, save_beliefs=true)

    p1 = plot(xlabel="Time", ylabel="Accumulated Evidence")
    for i in 1:2
        α = map(sim.bs) do b
            b.counts[i][1]
        end
        plot!(α)
    end
    #hline!([30], color=:black)

    p2 = plot(xlabel="Time", ylabel="Memory Strength")
    for i in 1:2
        μ, lo, hi = map(sim.bs) do b
            d = Beta(b.counts[i]...)
            lo, hi = quantile.(d, [0.1, 0.9])
            μ = mean(d)
            μ, μ - lo, hi - μ
        end |> invert
        plot!(μ, ribbon=(lo, hi))
    end

    for p in [p1, p2]
        vline!(p, findall(diff(sim.cs) .!= 0) .+ 1, color=:gray, lw=0.5)
    end

    plot(p1, p2, layout=(2,1), size=(400,500))
end

# %% --------
m = MetaMDP{2}(step_size=2, threshold=19, switch_cost=0., max_step=60)
pol = OptimalPolicy(m)
@show pol.V(initial_belief(m))
# %% --------
X = product(0:m.max_step-2, 0:m.threshold) do t, h
    t == m.max_step && return 0.
    h > t && return NaN
    b = Belief{2}(t, 1, ((1+h, 1+t-h), (1,1)))
    q1, q2 = Q(pol.V, b)
    #q1 ≈ q2 && return NaN
    q1 - q2
    #float(act(pol, b))
end

rng = maximum(filter(!isnan, (abs.(X))))
figure() do    
    heatmap(X', c=:RdBu_11, cbar=false, clims=(-rng, rng),
        grid=false, xlab="Time", ylab="Accumulated Evidence", size=(400,250)
    )
end

# %% --------
X = product(0:m.max_step-2, 0:m.threshold) do t, μ
    t == m.max_step && return 0.
    h > t && return NaN
    b = Belief{2}(t, 1, ((1+h, 1+t-h), (1, 1)))
    q1, q2 = Q(pol.V, b)
    #q1 ≈ q2 && return NaN
    float(act(pol, b))
end

figure() do    
    heatmap(X', c=:viridis, clim=(1, 2.5), cbar=false,
        grid=false,
        xlab="Time",
        ylab="Evidence",
    )
end