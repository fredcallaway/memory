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

# %% ==================== Policy Heatmap ====================

# m = MetaMDP{2}(sample_cost=.1, threshold=4, noise=2, switch_cost=.1,
#         max_step=30, prior=Normal(0.5, 2)
# )

# m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.1, threshold=4, noise=2, switch_cost=0,
#         max_step=30, prior=Normal(0.5, 2)
# )

m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.03, 
    threshold=10, noise=2, max_step=60, prior=Normal(0, 1)
)

B = BackwardsInduction(m; dv=0.2)

countmap([simulate(OptimalPolicy(B)).b.focused for i in 1:1000])

# %% --------
Q = B.Q[:, 1, :, 41, :, 1]
# X = Q[1, :, :] - Q[2, :, :]
# rng = maximum(filter(!isnan, (abs.(X))))
# figure() do
#     heatmap(X , c=:RdBu_11, cbar=false, clims=(-rng, rng))
# end

figure() do
    X = (Q[1, :, :] .> Q[2, :, :]) .* 2 .- 1
    X .*= .!((Q[1, :, :] .≈ Q[2, :, :]) .| isnan.(Q[1, :, :]))
    heatmap(X, c=:RdBu_11, cbar=false,
        xaxis="Time on Cue",
        yaxis="Accumulated Evidence"
    )
end





# %% ==================== Simulation ====================

m1 = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.1, 
    threshold=8, noise=2, max_step=60, prior=Normal(0.5, 2)
)

pol1 = OptimalPolicy(m1)

function sample_states(pol, N=10000)
    states = [Float64[] for i in 1:3]
    for i in 1:N
        s = sample_state(m)
        n_correct = (simulate(pol; s).b.focused == 1) + (simulate(pol; s).b.focused == 1)
        push!(states[n_correct + 1], s[1])
    end
    states
end

states = sample_states(pol1)

# figure() do
#     map(states) do ss
#         histogram!(ss)
#     end
# end

# %% --------

m = MetaMDP{2}(allow_stop=true, miss_cost=3, sample_cost=.1, switch_cost=.05,
    threshold=8, noise=2, max_step=60, prior=Normal(1.5, 2)
)

@time pol = OptimalPolicy(m)
# %% --------
@everywhere function make_frame(pol, N=10000; ms_per_sample=250, σ_duration=.2)
    sims = map(1:N) do i
        sim = simulate(pol; fix_log=FullFixLog())
        strength_first, strength_second = sim.s
        presentation_times = sim.fix_log.fixations .* float(ms_per_sample)
        presentation_times .+= (pol.m.switch_cost / pol.m.sample_cost) * ms_per_sample
        presentation_times .+= rand(Gamma(10, 10), length(presentation_times))
        # presentation_times = max.(1, rand.(Normal.(μ_pres, μ_pres .* σ_duration)))
        outcome = sim.b.focused

        (;strength_first, strength_second, presentation_times, outcome,
         duration_first = sum(presentation_times[1:2:end]),
         duration_second = sum(presentation_times[2:2:end]))
    end
    DataFrame(sims[:])
end

df = make_frame(pol)
@show counts(length.(df.presentation_times)) ./ 10000
@show mean(sum.(df.presentation_times))
# df |> CSV.write("results/sim_gaussian2.csv")
@show mean(df.outcome .== 1)





# %% ==================== Search ====================



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



