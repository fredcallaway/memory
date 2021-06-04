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
