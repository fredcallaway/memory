@everywhere include("binomial_accumulator.jl")
@everywhere include("utils.jl")
using ProgressMeter

# %% --------
@everywhere N_SIM = 10000

policies = map(50:25:100) do miss_cost
    m = MetaMDP{1}(;step_size=4, max_step=100, threshold=100, sample_cost=1, miss_cost, allow_stop=true)
    V = ValueFunction(m); V(initial_belief(m))
    OptimalPolicy(V)
end

all_res = @showprogress pmap(Iterators.product(policies, 0:.01:0.5)) do (pol, p)
    map(1:N_SIM) do i
        sim = simulate(pol, s=(p,))
        (;pol.m.miss_cost, p, n_step = length(sim.cs), 
         success = sim.bs[end].focused == 1, 
         give_up = sim.cs[end] == 0)
    end
end

using CSV
flatten(all_res) |> CSV.write("results/stopping_sim.csv")

# %% --------


# using DataFrames

