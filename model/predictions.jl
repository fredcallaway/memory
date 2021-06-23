using ProgressMeter
@everywhere begin
    using CSV
    using DataFrames
    include("binomial_accumulator.jl")
    include("utils.jl")
end
include("figure.jl")

# %% --------

@everywhere function make_sims(pol, N=10000)
    time_limit = 15
    ms_per_sample = 1000 * (time_limit / pol.m.max_step)
    b = mutate(initial_belief(pol.m), focused=1)
    sims = map(1:N) do i
        s = (s1, s2) = sample_state(m)
        sim = simulate(pol; s, b)
        presentation_times = parse_presentations(sim.cs, ms_per_sample)
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

m = MetaMDP(step_size=4, max_step=60, threshold=20, sample_cost=1, switch_cost=6, miss_cost=0, prior=(2,6))

monte_carlo() do 
    p = rand(Beta(m.prior...))
    m.max_step * p * m.step_size > m.threshold
end |> print

V = ValueFunction(m)
@time V(initial_belief(m))
df = make_sims(SoftOptimalPolicy(V; β=0.3))
@show mean(length.(df.presentation_times))
CSV.write("results/sim_optimal_prior.csv", df)


# %% --------
include("empirical_fixation.jl")
df = make_sims(empirical_policy(m))
countmap(df.outcome)
@show mean(length.(df.presentation_times))
CSV.write("results/sim_empirical.csv", df)

# %% --------

df = make_sims(empirical_commitment_policy2(m))
countmap(df.outcome)
@show mean(length.(df.presentation_times))
CSV.write("results/sim_empirical_commitment.csv", df)
