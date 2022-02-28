
# %% ==================== Benchmarking ====================
using BenchmarkTools
using TimerOutputs
const to = TimerOutput()
# %% --------

prm = opt_prm

@time pre_pol, crit_pol = optimal_policies(prm)

@btime simulate(crit_pol)

@btime simulate_exp2(pre_pol, crit_pol, 10000);

# %% --------

function simulate_exp2(pre_pol, crit_pol, N=1000000; 
                       strength_drift=Normal(0, 1e-9), duration_noise=Gamma(1e-9,1e-9))
    @timeit to "sample" begin
        strengths = sample_strengths(pre_pol,  2N; strength_drift)
        pairs = map(1:2:2N) do i
            s1, pretest_accuracy_first = strengths[i]
            s2, pretest_accuracy_second = strengths[i+1]
            (;pretest_accuracy_first, pretest_accuracy_second), (s1, s2)
        end
    end

    map(pairs) do (pretest_accuracy, s)
        @timeit to "simulate" begin
            sim = simulate(crit_pol; s, fix_log=FullFixLog())
        end
        @timeit to "post" begin
            presentation_times = sim.fix_log.fixations .* float(ms_per_sample)
            # presentation_times .+= (crit_pol.m.switch_cost / crit_pol.m.sample_cost) * ms_per_sample
            presentation_times .+= rand(duration_noise, length(presentation_times))
            (;response_type = sim.b.focused == -1 ? "empty" : "correct",
              choose_first = sim.b.focused == 1,
              pretest_accuracy..., 
              strength_first = s[1], strength_second = s[2],
              presentation_times
             )
        end
    end |> DataFrame
end

simulate_exp2(pre_pol, crit_pol, 10)
reset_timer!(to)
simulate_exp2(pre_pol, crit_pol, 100000)
to

# %% ==================== From experiment 1 ====================

exp1_prm = deserialize("tmp/exp1_fit_prm2")
prm = (;exp1_prm..., switch_cost=0.)
df = simulate_exp2(optimal_policies, prm)
# target = critical_metrics(trials)
# x = critical_metrics(make_trials(df))
# x.nfix
# target.nfix

df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% ==================== Old ====================



# %% --------
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



# %% ==================== Manual ====================

# prm = (drift_μ = 0.6, drift_σ = 0.8, threshold = 5, sample_cost = 0.04, noise = 1.3, switch_cost=.01)
# prm = (drift_μ = 0.5, drift_σ = 0.6, threshold = 4, sample_cost = 0.01, noise = 1.0, switch_cost=.01)
prm = (drift_μ = 0.9, drift_σ = 1.1, threshold = 7, sample_cost = 0.002, 
    noise = 1.4, switch_cost=.002, strength_drift_μ=0, strength_drift_σ=1e-9)

@time df = simulate_optimal(prm)

# counts(make_trials(df).n_pres, 1:4) ./ nrow(df)
# counts(trials.n_pres, 1:4) ./ nrow(trials)
df |> make_trials |> CSV.write("results/exp2/optimal_trials.csv")
df |> make_fixations |> CSV.write("results/exp2/optimal_fixations.csv")

# %% ==================== Fit ====================

@everywhere function twocue_metrics(trials, fixations)
    trials = @rsubset(trials, :response_type == "correct")
    fixations = @rsubset(fixations, :response_type == "correct")

    avg_ptime = @chain fixations begin
        @rsubset(:presentation != :n_pres)
        @by(:wid, :mean=mean(:duration), :sd=std(:duration))
    end

    z_durations = @chain fixations begin
        leftjoin(avg_ptime, on=:wid)
        @rsubset(!isnan(:sd))
        @rtransform(:rel_acc = :pretest_accuracy_first - :pretest_accuracy_second)
        @rsubset(:presentation <= 3 && :presentation != :n_pres)
        @rtransform(:duration_z = (:duration - :mean) / :sd)
        @by([:presentation, :rel_acc], :x = mean(:duration_z))
        wrapdims(:x, :presentation, :rel_acc)
        sortkeys
    end
    (;
        z_durations,
         nfix_hist = counts(trials.n_pres, 1:4) ./ nrow(trials),
         p_correct = mean(trials.response_type .== "correct"),
    )
end
@everywhere twocue_metrics(df) = compute_metrics(make_trials(df), make_fixations(df))

target = compute_metrics(trials, fixations);
@everywhere target = $target

@everywhere squared(x) = x^2
@everywhere function loss(pred)
    nfix_loss = crossentropy(target.nfix_hist, pred.nfix_hist)
    isfinite(nfix_loss) || return Inf
    duration_loss = mean(squared.(pred.z_durations .- target.z_durations))
    nfix_loss + duration_loss
end