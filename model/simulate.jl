using DataFrames
include("binomial_accumulator.jl")
include("utils.jl")
include("empirical_fixation.jl")
include("constants.jl")
# %% --------

function make_sims(pol, N=10000)
    ms_per_sample = 1000 * (TIME_LIMIT / pol.m.max_step)
    b = mutate(initial_belief(pol.m), focused=1)
    sims = map(1:N) do i
        s = (s1, s2) = sample_state(pol.m)
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

function simulate_individual_empirical(m::MetaMDP, N=100; kws...)
    pols = individual_empirical_policies(m; kws...)
    filter!(pols) do pol
        length(pol.commitment_dist.support) != 0
    end;

    mapreduce(vcat, pols) do pol
        make_sims(pol, N)
    end
end