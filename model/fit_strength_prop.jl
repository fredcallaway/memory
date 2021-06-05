using GLM

# start with predicions.jl
@everywhere zscore(x) = (x .- mean(x)) ./ std(x)

@everywhere function strength_prop_coef(df)
    df.rel_strength = zscore(df.strength_first - df.strength_second)
    df.prop_first = df.duration_first ./ (df.duration_first .+ df.duration_second)
    model = lm(@formula(prop_first ~ rel_strength), df)
    coef(model)[2]
end

opt_coef = strength_prop_coef(CSV.read("results/sim_optimal.csv", DataFrame))

# %% ==================== Search space ====================

using Sobol
include("box.jl")
box = Box(
    lo = (0, 1),
    hi = (0, 1),
    threshold = (50, 200),
)
xs = Iterators.take(SobolSeq(n_free(box)), 5000) |> collect
prms = map(xs) do x
    prm = box(x)
    prm.hi - prm.lo > .1 || return missing
    (;prm..., threshold=round(Int, prm.threshold))
end |> skipmissing |> unique
# %% --------

@everywhere function random_sims(;lo, hi, threshold, length=300)
    m = MetaMDP(;step_size=4, max_step, threshold, sample_cost=0, switch_cost=0, miss_cost=0)
    pol = SwitchDistributionPolicy(m, empirical_fix_dist)
    make_sims(pol, range(lo, hi; length))
end

@everywhere function objective(;lo, hi, threshold)
    strength_prop_coef(random_sims(;lo, hi, threshold))
end

res = @showprogress pmap(prms) do prm
    objective(;prm...)
end

best = prms[argmax(res)]
@show objective(;prms[argmax(res)]...)
# %% --------

m = MetaMDP(step_size=4, max_step, best.threshold, sample_cost=1, switch_cost=4, miss_cost=0)
df = random_sims(;best..., length=200)
df |> CSV.write("results/sim_rand_gamma.csv")
strength_prop_coef(df)



