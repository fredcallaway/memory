@everywhere begin
    using StatsBase
    include("common.jl")
    # include("empirical_fixation.jl")
end

# time out proportion
# average fixation duration
# average fixation length

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)

pretest = CSV.read("../data/processed/exp2/pretest.csv", DataFrame, missingstring="NA")
trials = CSV.read("../data/processed/exp2/trials.csv", DataFrame, missingstring="NA")
fixations = CSV.read("../data/processed/exp2/fixations.csv", DataFrame, missingstring="NA")

pretest = @rsubset pretest :practice == false :block == 3
@everywhere trials = $trials
@everywhere fixations = $fixations
# trials = @rsubset(all_trials, :response_type == "correct")
# fixations = @rsubset(all_fixations, :response_type == "correct")

# %% ==================== Simulate ====================

@everywhere function make_frame(pol1, pol2, N=10000; ms_per_sample=200)
    strengths = sample_strengths(pol1,  2N)
    pairs = map(1:2:2N) do i
        s1, pretest_accuracy_first = strengths[i]
        s2, pretest_accuracy_second = strengths[i+1]
        (;pretest_accuracy_first, pretest_accuracy_second), (s1, s2)
    end

    sims = map(pairs) do (pretest_accuracy, s)
        sim = simulate(pol2; s, fix_log=FullFixLog())
        presentation_times = sim.fix_log.fixations .* float(ms_per_sample)
        # presentation_times .+= (pol2.m.switch_cost / pol2.m.sample_cost) * ms_per_sample
        presentation_times .+= rand(Gamma(10, 10), length(presentation_times))
        (;response_type = sim.b.focused == -1 ? "empty" : "correct",
          choose_first = sim.b.focused == 1,
          pretest_accuracy..., 
          strength_first = s[1], strength_second = s[2],
          presentation_times
         )
    end
    DataFrame(sims[:])
end

@everywhere function make_trials(df)
    safeindex(x, i) = length(x) < i ? NaN : x[i]
    @chain df begin
         @rtransform(
            :first_pres_time = safeindex(:presentation_times, 1),
            :second_pres_time = safeindex(:presentation_times, 2),
            :third_pres_time = safeindex(:presentation_times, 3),
            :last_pres_time = :presentation_times[end],
            :n_pres = length(:presentation_times),
            :total_first = sum(:presentation_times[1:2:end]),
            :total_second = sum(:presentation_times[2:2:end]),
            :wid = "optimal"
        )
        @transform :rt = :total_first .+ :total_second
        select(setdiff(names(trials)))
    end
end

@everywhere function make_fixations(df)
    @chain df begin
        @transform(:trial_id = 1:nrow(df))
        @rtransform(:n_pres = length(:presentation_times))
        @rtransform(:presentation = 1:(:n_pres), :wid = "optimal")
        DataFrames.flatten([:presentation_times, :presentation])
        DataFrames.rename(:presentation_times => :duration)
        select(setdiff(names(fixations)))
    end
end

@everywhere function simulate_optimal(prm::NamedTuple, N=100000)
    m1 = MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.sample_cost, prm.switch_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
    pol1 = OptimalPolicy(m1; dv=m1.threshold*.01)

    m2 = MetaMDP{2}(;allow_stop=true, max_step=60, miss_cost=2,
        prm.threshold, prm.sample_cost, prm.switch_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
    pol2 = OptimalPolicy(m2; dv=m2.threshold*.01)
    make_frame(pol1, pol2, N)
end

# %% ==================== Tune pretest ====================

@everywhere function simulate_pretest(prm, N=10000)
    m = MetaMDP{1}(;allow_stop=true, max_step=60, miss_cost=1,
        prm.threshold, prm.sample_cost, prm.noise,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
    pol = OptimalPolicy(m; dv=m.threshold*.01)
    
    mapreduce(vcat, 1:N) do i
        s = sample_state(pol.m)
        map(1:2) do j
            sim = simulate(pol; s, fix_log=RTLog())
            (;
                wid="optimal",
                word=i,
                strength=only(s),
                response_type = sim.b.focused == -1 ? "empty" : "correct",
                rt=sim.fix_log.rt * ms_per_sample,
            )
        end
    end |> DataFrame
end

@everywhere function pretest_metrics(df)
    acc_rate = @chain df begin
        @by([:word, :wid], :x=mean(:response_type .== "correct"))
        @by(_, :x, :n=length(:x) ./ nrow(_))
        wrapdims(:n, :x)
        sortkeys
    end

    rt_μ, rt_σ = @chain df begin
        @rsubset :response_type .== "correct"
        @with mean(:rt), std(:rt)
    end

    (;acc_rate, rt_μ, rt_σ)
end

target = pretest_metrics(pretest)

prms = grid(
    drift_μ=0:.1:.6,
    drift_σ=0.8:.1:1.2,
    threshold=5:9,
    sample_cost=.04:.01:.08,
    noise=1.3:.1:1.7
)

metrics = @showprogress pmap(prms) do prm
    pretest_metrics(simulate_pretest(prm))
end

# %% --------
L = map(metrics) do pred
    10sum(squared.(target.acc_rate .- pred.acc_rate)) +
    # squared((target.rt_σ - pred.rt_σ) / 1000) +
    # 5squared((target.rt_μ - pred.rt_μ) / 1000) +
    0
end

prms[argmin(L)]
println(metrics[argmin(L)])
println(target)



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


# %% ==================== Manual ====================

prm = (drift_μ = 0.6, drift_σ = 0.8, threshold = 5, sample_cost = 0.04, noise = 1.3, switch_cost=.01)

df = simulate_optimal(prm)
# %% --------

twocue_metrics(df).z_durations

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



