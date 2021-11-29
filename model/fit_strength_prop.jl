using CSV
using ProgressMeter
@everywhere include("simulate.jl")
@everywhere using GLM
@everywhere zscore(x) = (x .- mean(x)) ./ std(x)

@everywhere function strength_prop_coef(df)
    df = filter(df) do row
        length(row.presentation_times) > 1
    end
    size(df, 1) < 10 && return 0
    df.rel_strength = zscore(df.strength_first - df.strength_second)
    df.prop_first = df.duration_first ./ (df.duration_first .+ df.duration_second)
    model = lm(@formula(prop_first ~ rel_strength), df)
    coef(model)[2]
end

opt_sim = CSV.read("results/sim_optimal.csv", DataFrame)
@time opt_coef = strength_prop_coef(opt_sim)

# %% ==================== Search space ====================

prms = grid(
    α = 1:10,
    β = 1:10,
    threshold = [1:9; 10:5:50],
)

# %% --------

@everywhere function random_sims(;α, β, threshold)
    m = MetaMDP(; threshold, prior=(α, β), step_size=4, max_step=60, sample_cost=0, switch_cost=0, miss_cost=0)
    simulate_individual_empirical(m; commitment=false)
end

@everywhere function objective(;α, β, threshold)
    strength_prop_coef(random_sims(;α, β, threshold))
end

res = @showprogress pmap(prms) do prm
    objective(;prm...)
end
# %% --------

objective(α=1, β=1, threshold=30)

objective(;prms[argmax(res)]...)

objective(α=10, β=1, threshold=1)

sim = random_sims(α=1, β=1, threshold=10)
#sim = random_sims(;prms[argmax(res)]...)
objective(α=1, β=1, threshold=10)
sim |> CSV.write("results/sim_rand_fit.csv")


# %% --------
best = prms[argmax(res)]
@show objective(;prms[argmax(res)]...)
# %% --------

m = MetaMDP(step_size=4, max_step, best.threshold, sample_cost=1, switch_cost=4, miss_cost=0)
df = random_sims(;best..., length=200)
df |> CSV.write("results/sim_rand_gamma.csv")
strength_prop_coef(df)

# %% --------


# %% ==================== Version 2 ====================

@everywhere using GLM
@everywhere function strength_prop_coef(df)
    df = filter(df) do row
        length(row.presentation_times) >= 2
    end
    size(df, 1) < 10 && return 0
    df.rel_strength = zscore(collect(skipmissing(df.strength_first - df.strength_second)))
    df.prop_first = df.duration_first ./ (df.duration_first .+ df.duration_second)
    model = lm(@formula(prop_first ~ rel_strength), df)
    coef(model)[2]
end

# %% --------
@everywhere include("empirical_fixation.jl")
strength_prop_coef(make_frame(random_policy(m)))

@everywhere function random_strength(prm)
    @chain prm begin
        make_mdp
        random_policy(commitment=false)
        make_frame
        strength_prop_coef
    end
end

rstrength = @showprogress pmap(random_strength, prms)
serialize("results/rstrength", rstrength)

# strength_prop_coef(@rsubset df !ismissing(:strength_second) && !ismissing(:strength_first))

# %% ==================== Analyze ====================

rstrength = deserialize("results/rstrength")

# %% --------

R = rstrength(step_size=4, max_step=120)
best = keymin(R)
figure() do
    R(;best.threshold, best.switch_cost) |> heatmap
end

figure() do
    R(;best.α, best.β) |> heatmap
end

