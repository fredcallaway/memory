


# %% ==================== Vector beliefs ====================
include("mdp.jl")
include("utils.jl")
include("figure.jl")
# %% --------

function plot_vec!(h, t, c)
    scatter!([t], [h]; c, markersize=10)
    lo, hi = (6-t) .* quantile(Beta(1+h,1+t-h), [0.1, 0.9])
    @show lo hi
    plot!([t, 6], [h, h+lo]; c, fillrange=[h, h+hi], fillalpha=0.3, alpha=0)
    hline!([6], line=(:black,))
end
# function plot_vec!(h, t, c)
#     plot!([0, t], [0, h]; c)
#     lo, hi = t .* quantile(Beta(1+h,1+t-h), [0.1, 0.9])
#     plot!([0, t], [0, lo]; c, fillrange=[0, hi], fillalpha=0.3, alpha=0)
# end

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


# %% ==================== Outdated ====================
include("binomial_accumulator.jl")
include("figure.jl")
using StatsPlots

# %% --------
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

# %% ==================== Policy Heatmap ====================

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
        grid=false, xlab="Time on Cue", ylab="Accumulated Evidence", size=(400,250)
    )
end


