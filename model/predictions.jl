using ProgressMeter
@everywhere begin
    using CSV
    using GLM
    using DataFrames
    include("binomial_accumulator.jl")
    include("utils.jl")
end
include("figure.jl")

# %% --------

@everywhere function make_sims(pol, strengths=0:0.01:1, reps=1)
    G = Iterators.product(strengths, strengths, 1:reps)
    b = mutate(initial_belief(pol.m), focused=1)
    sims = map(G) do (s1, s2)
        s = (s1, s2)
        sim = simulate(pol; s, b)
        presentation_times = parse_presentations(sim.cs)
        # recode things to make 1 correspond to first seen item
        outcome = let x = sim.bs[end].focused
            x == -1 ? -1 :
            x == first(sim.cs) ? 1 :
            2
        end
        strength_first, strength_second = first(sim.cs) == 1 ? (s1, s2) : (s2, s1)


        (;strength_first, strength_second, presentation_times, outcome,
         duration_first = sum(presentation_times[1:2:end]),
         duration_second = sum(presentation_times[2:2:end]))
    end
    DataFrame(sims[:])
end

# %% --------
m = MetaMDP(step_size=4, max_step=60, threshold=100, sample_cost=1, switch_cost=4, miss_cost=0)
V = ValueFunction(m)
@time V(initial_belief(m))
make_sims(SoftOptimalPolicy(V; β=0.3)) |> CSV.write("results/sim_optimal.csv")

# %% --------

m = MetaMDP(step_size=4, max_step=60, threshold=100, sample_cost=1, switch_cost=4, miss_cost=0, prior=(1,6))
V = ValueFunction(m)
@time V(initial_belief(m))
make_sims(SoftOptimalPolicy(V; β=0.3)) |> CSV.write("results/sim_optimal_prior.csv")


# %% ==================== Fit random policy ====================
@everywhere zscore(x) = (x .- mean(x)) ./ std(x)

@everywhere function strength_prop_coef(df)
    df.rel_strength = zscore(df.strength_first - df.strength_second)
    df.prop_first = df.duration_first ./ (df.duration_first .+ df.duration_second)
    model = lm(@formula(prop_first ~ rel_strength), df)
    coef(model)[2]
end

opt_coef = strength_prop_coef(CSV.read("results/sim_optimal.csv", DataFrame))

# %% ==================== Empirical fixation distribution ====================

version = "v5.0"
multi = CSV.read("../data/$version/multi-recall.csv", DataFrame)
using JSON
multi.presentation_times = Vector{Vector{Float64}}(map(JSON.parse, multi.presentation_times))

max_step = 60
ms_per_sample = 1000 * (15 / max_step)
x = reduce(vcat, multi.presentation_times) ./ 250
empirical_fix_dist = fit(Gamma, x)

@everywhere begin
    max_step = $max_step
    empirical_fix_dist = $empirical_fix_dist
end

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




