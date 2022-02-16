using ProgressMeter
using JSON
@everywhere begin
    using CSV, DataFrames, DataFramesMeta
    using StatsBase
    include("utils.jl")
    include("figure.jl")
    include("constants.jl")
    include("mdp.jl")
    include("optimal_policy.jl")
    include("empirical_fixation.jl")
end

# time out proportion
# average fixation duration
# average fixation length

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)

trials = CSV.read("../data/processed/exp2/trials.csv", DataFrame)

# %% --------

@chain trials begin
    @subset(:n_pres .> 1)
    @by(:pre_correct_first, :y=mean(:first_pres_time))
    @orderby(:pre_correct_first)
end

# %% --------



# %% ==================== Sample states ====================

m1 = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.06, 
    threshold=7, noise=1.5, max_step=60, prior=Normal(0, 1)
)

pol1 = OptimalPolicy(m1)

function sample_states(pol, N=10000)
    states = [Float64[] for i in 1:3]
    for i in 1:N
        s = sample_state(pol.m)
        n_correct = (simulate(pol; s).b.focused == 1) + (simulate(pol; s).b.focused == 1)
        push!(states[n_correct + 1], s[1])
    end
    states
end

states = sample_states(pol1)

# %% --------

m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.06, switch_cost=.06,
    threshold=7, noise=1.5, max_step=60, prior=Normal(0, 1)
)

@time B = BackwardsInduction(m; dv=0.2)
pol = OptimalPolicy(B)


function make_frame(pol, N=10000; ms_per_sample=200)
    states = sample_states(pol, 2N)
    ss = mapreduce(vcat, 0:0.5:1, states) do pre_correct, strengths
        map(strengths) do s
            (pre_correct, s)
        end
    end
    shuffle!(ss)
    pairs = map(1:2:2N) do i
        pre_correct_first, s1 = ss[i]
        pre_correct_second, s2 = ss[i+1]
        (;pre_correct_first, pre_correct_second), (s1, s2)
    end

    sims = map(pairs) do (pre_correct, s)
        sim = simulate(pol; s, fix_log=FullFixLog())
        presentation_times = sim.fix_log.fixations .* float(ms_per_sample)
        presentation_times .+= (pol.m.switch_cost / pol.m.sample_cost) * ms_per_sample
        presentation_times .+= rand(Gamma(10, 10), length(presentation_times))
        # presentation_times = max.(1, rand.(Normal.(μ_pres, μ_pres .* σ_duration)))
        outcome = sim.b.focused


        (;response_type = outcome == -1 ? "empty" : "correct",
          choose_first = outcome == 1,
          pre_correct..., 
          presentation_times
         )
    end
    DataFrame(sims[:])
end

duration_first = sum(presentation_times[1:2:end])
duration_second = sum(presentation_times[2:2:end])

df = make_frame(pol)
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



