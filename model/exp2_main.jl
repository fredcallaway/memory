@time @everywhere include("common.jl")
mkpath("results/exp2")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
# %% --------

pretest = CSV.read("../data/processed/exp2/pretest.csv", DataFrame, missingstring="NA")
trials = CSV.read("../data/processed/exp2/trials.csv", DataFrame, missingstring="NA")
fixations = CSV.read("../data/processed/exp2/fixations.csv", DataFrame, missingstring="NA")

pretest = @rsubset pretest :practice == false :block == 3
trials = @rsubset trials :n_pres > 0

@everywhere trials = $trials
@everywhere fixations = $fixations

# %% ==================== Fit pretest ====================

pre_prms = grid(10, Box(
    drift_μ = (0, 1),
    drift_σ = (0.5, 1.5),
    threshold = (1, 10),
    sample_cost = (.001, .05, :log),
    noise = (.2, 2),
))

mkpath(".cache/exp2_pre_metrics")
pre_metrics = @showprogress pmap(pre_prms) do prm
    cache(".cache/exp2_pre_metrics/$(stringify(prm))") do
        pretest_metrics(simulate_pretest(prm))
    end
end

# %% --------

pre_target = pretest_metrics(pretest)

L = map(pre_metrics) do pred
    sum(squared.(pre_target.acc_rate .- pred.acc_rate)) +
    squared((pre_target.rt_μ - pred.rt_μ) / 3pre_target.rt_σ) +
    squared((pre_target.rt_σ - pred.rt_σ) / 3pre_target.rt_σ)
end;

# L = map(pre_metrics) do pred
#     sum(squared.(pre_target.acc_rate .- pred.acc_rate)) +
#     squared((pre_target.rt_μ - pred.rt_μ) / 1000) +
#     squared((pre_target.rt_σ - pred.rt_σ) / 1000) +
#     0
# end;

flat_prms = collect(pre_prms)[:];
flat_L = collect(L)[:];
flat_prms[sortperm(flat_L)]
pre_prm = keymin(L)

# %% ==================== Fit critical ====================

@everywhere include("simulate_exp2.jl")

@everywhere function critical_metrics(trials)
    nfix = counts(trials.n_pres, 1:4) ./ nrow(trials)
    push!(nfix, mean(trials.n_pres .> 4))
    accuracy = mean(trials.response_type .== "correct")

    choice_rate = @chain trials begin
        @rsubset :response_type == "correct"
        @by :pretest_accuracy_first :choice = mean(skipmissing(:choose_first))
        wrapdims(:choice, :pretest_accuracy_first)
        sortkeys
    end

    rt_μ, rt_σ = @chain trials begin
        @rsubset :response_type .== "correct"
        @with (mean(:rt), std(:rt))
    end
    (;nfix, accuracy, choice_rate, rt_μ, rt_σ)
end

crit_prms = grid(10, Box(
    switch_cost = (.0001, .01, :log),
    strength_drift_μ = (-.5, 0),
    strength_drift_σ = (1e-3, .5)
))

mkpath(".cache/exp2_crit_metrics")
crit_metrics = @showprogress pmap(crit_prms) do cprm
    prm = (;pre_prm..., cprm...)
    cache(".cache/exp2_crit_metrics/$(stringify(prm))") do
        trials = make_trials(simulate_optimal(prm))
        GC.gc()
        critical_metrics(trials)
    end
end

# crit_metrics = deserialize("tmp/crit_metrics")
# %% --------
crit_target = critical_metrics(trials)

L = map(crit_metrics) do pred
    mean(squared.(crit_target.nfix .- pred.nfix)) +
    mean(squared.(pred.choice_rate .- crit_target.choice_rate)) +
    mean(squared((pred.rt_μ - crit_target.rt_μ)/3crit_target.rt_σ))
end

prm = (;prm_pre..., keymin(L)...) #TODO run this 

# %% --------

# prm = (drift_μ = 0.9, drift_σ = 1.1, threshold = 7, sample_cost = 0.002, noise = 1.4, switch_cost = 0.0002, strength_drift_μ = -0.1, strength_drift_σ = 0.4)
# prm = (drift_μ = 0.9, drift_σ = 1.1, threshold = 7, sample_cost = 0.002, noise = 1.4, switch_cost = 0.0002, strength_drift_μ = 0., strength_drift_σ = 1e-9)
df = simulate_optimal(prm)
df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% ====================  ====================

m1 = MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
    prm.threshold, prm.sample_cost, prm.switch_cost, prm.noise,
    prior=Normal(prm.drift_μ, prm.drift_σ),
)
pol1 = OptimalPolicy(m1; dv=m1.threshold*.02)

m2 = MetaMDP{2}(;allow_stop=true, max_step=60, miss_cost=2,
    prm.threshold, prm.sample_cost, prm.switch_cost, prm.noise,
    prior=Normal(prm.drift_μ, prm.drift_σ),
)
pol2 = OptimalPolicy(m2; dv=m2.threshold*.02)

# %% --------



df = make_frame(pol1, pol2, 10N; 
    duration_noise=Gamma(1e-3, 1e-3), strength_drift=Normal(0, 1e-3))

mean(df.response_type .== "correct")
counts(make_trials(df).n_pres, 1:4) ./ nrow(df)

df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% ==================== Run ====================

mkpath("tmp/sims")

prms = grid(
    drift_μ=0:.1:.6,
    drift_σ=0.8:.1:1.2,
    threshold=5:9,
    sample_cost=.04:.01:.08,
    switch_cost=[.005, .01, .02, .03],
    noise=1.3:.1:1.7
)

# %% ==================== One go ====================

L = @showprogress pmap(prms) do prm
    prm |> simulate_optimal |> compute_metrics |> loss
end

# %% --------
prms[argmin(L)]
minimum(L)
df = simulate_optimal(prms[argmin(L)]);
pred = compute_metrics(df)
loss(pred)

# %% --------

prm = (drift_μ = 0.0, drift_σ = 1, threshold = 5, sample_cost = 0.05, switch_cost = 0.00, noise = 1.5)
df = simulate_optimal(prm, 10000);
pred = compute_metrics(df)
pred.z_durations
pred.nfix_hist


df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% ==================== Separate steps  ====================


sims = @showprogress map(simulate_optimal, prms)
predictions = map(compute_metrics, sims);
target = compute_metrics(trials, fixations);

L = map(loss, predictions)

prms[argmin(L)]
predictions[argmin(L)].nfix_hist

df = simulate_optimal(prms[argmin(L)])
loss(compute_metrics(df))

# %% --------
target.nfix_hist

df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% --------


figure() do
   plot!(second_fixation_duration(make_trials(df), make_fixations(df)))
   plot!(second_fixation_duration(trials, fixations))
end      

prm = first(prms)
second_fixation_duration(make_trials(df), make_fixations(df))

# %% --------


# %% ==================== Hand selected ====================

m1 = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.06, 
    threshold=7, noise=1.5, max_step=60, prior=Normal(0, 1)
)
pol1 = OptimalPolicy(m1)

# %% --------
m2 = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.06, switch_cost=.06,
    threshold=7, noise=1.5, max_step=60, prior=Normal(0, 1)
)
# note: was a bug here with m instead of m2
@time B = BackwardsInduction(m2; dv=0.2)
pol2 = OptimalPolicy(B)

# %% --------
mkpath("results/exp2")
df = make_frame(pol1, pol2)
df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

pol_rand = random_policy(m, commitment=false)
df_rand = make_frame(pol1, pol_rand)
df_rand |> make_trials |> CSV.write("results/exp2/random_trials.csv")
df_rand |> make_fixations |> CSV.write("results/exp2/random_fixations.csv")

df
# %% ==================== OLD ====================


# %% --------

@show counts(length.(df.presentation_times)) ./ 10000
@show mean(sum.(df.presentation_times))
@show mean(df.outcome .!= -1)

df |> CSV.write("results/exp2_optimal.csv")

make_frame(random_policy(m, commitment=false)) |> CSV.write("results/exp2_random.csv")

# %% ==================== Search ====================




# %% --------


# %% --------
# m = MetaMDP(step_size=4, max_step=120, threshold=20, sample_cost=1, switch_cost=5, miss_cost=0, prior=(2, 6))
# pol = SoftOptimalPolicy(m, 0.3)
# make_frame(pol, 1000; ms_per_sample=100) |> CSV.write("results/sim_sanity_optimal.csv")
# include("empirical_fixation.jl")
# @chain random_policy(m; commitment=false) make_frame(1000; ms_per_sample=250) CSV.write("results/sim_sanity_rand.csv", _)
# %% --------

prms = grid(
    step_size=[4],
    max_step=[120],
    miss_cost=[0,100],
    prior=[(1,1), (2,6)],
    # prior=[(1,1), (2,4), (2,6)],
    # threshold = 30:15:60,
    threshold = [60],
    switch_cost = [1, 2,5],
    invtemp = [1],
)

@everywhere make_mdp(prm) = MetaMDP(;prm.step_size, prm.max_step, prm.threshold,
                             prm.switch_cost, prm.miss_cost, prm.prior)

dfs = @showprogress pmap(prms) do prm
    m = make_mdp(prm)
    pol = SoftOptimalPolicy(m, prm.invtemp)
    sim = make_frame(pol, 5000; ms_per_sample=100)
    for (k, v) in pairs(prm)
        if k == :prior
            v = string(v)
        end
        sim[!, k] .= v
    end
    sim
end

for (prm, df) in zip(prms, dfs)
    df[!, :wid] .= string(hash(prm), base=62)
end

df = reduce(vcat, dfs[:]) 
size(df, 1)
df |> CSV.write("results/sim_many_optimal.csv")

# %% --------

m = MetaMDP{2}(step_size=4, max_step=120, threshold=40, sample_cost=1, switch_cost=3, miss_cost=100, prior=(1, 1))
@time pol = SoftOptimalPolicy(m, 1) # 1.3 seconds

value(pol.B, initial_belief(m)) # -27.3027
df = make_frame(pol, 50000) 
@with(df, :duration_first + :duration_second) |> mean

df |> CSV.write("results/sim_new_optimal.csv")

# %% --------
pol2 = random_policy(mutate(m; switch_cost=0), commitment=false)
df2 = make_frame(pol2, 50000)
@with(df2, :duration_first + :duration_second) |> mean
# FIXME: this model is getting free accumulation b/c it doesn't pay swtich cost
# the switch cost should be accounted for when parsing fixation durations somehow...

df2 |> CSV.write("results/sim_new_random.csv")

# %% ==================== Explore space ====================

m = MetaMDP(step_size=4, max_step=120, threshold=20, sample_cost=1, switch_cost=5, miss_cost=0, prior=(2, 6))
pol = SoftOptimalPolicy(m, 0.3)
make_frame(pol, 1000; ms_per_sample=125) |> CSV.write("results/sim_sanity_optimal.csv")

# %% ==================== Old  ====================


pol = OptimalPolicy(make_mdp(best))
try_one(best).μ_rt
target.μ_rt
sim = make_frame(pol)
sim 
@chain sim begin
    @rsubset :outcome != -1
    @with mean(:duration_first .+ :duration_second)
end
sim |> CSV.write("results/sim_new_optimal.csv")



