using Revise
using CSV
using DataFrames
includet("binomial_accumulator.jl")
include("figure.jl")
includet("utils.jl")

# %% --------

m = MetaMDP(step_size=4, max_step=60, threshold=100, sample_cost=1, switch_cost=4, miss_cost=0)


# %% --------

function make_sims(pol)
    G = Iterators.product(0:0.01:1, 0:0.01:1)
    b = mutate(initial_belief(m), focused=1)
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

df = make_sims(RandomSwitchPolicy(m, 0.03))
df |> CSV.write("results/sim_random.csv")

# %% --------

V = ValueFunction(m)
@time V(initial_belief(m))
make_sims(SoftOptimalPolicy(V; β=0.3)) |> CSV.write("results/sim_optimal.csv")

