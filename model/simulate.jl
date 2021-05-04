using Revise
using ProgressMeter
@everywhere using TypedTables
using SplitApplyCombine
using CSV

@everywhere include("binomial_accumulator.jl")
@everywhere include("utils.jl")

mkpath("tmp")
mkpath("tmp/V")
mkpath("tmp/sim")
mkpath("results")
# %% --------

G = grid(step_size=[4], max_step=[60], difficulty=.4:.05:.6, switch_cost=[0, 2, 4], miss_cost=[0, 600])

@everywhere function make_mdp(g)
    @unpack step_size, max_step, difficulty, switch_cost, miss_cost = g
    threshold = step_size * max_step * difficulty
    MetaMDP(;step_size, max_step, threshold, switch_cost, miss_cost)
end
# %% --------

@everywhere function generate_sims(policy::Policy)
    sims = NamedTuple[]    
    foreach(.3:.1:.6) do v0
        foreach(0:.01:.2) do vd
            s = (v0 - vd, v0 + vd)
            foreach(1:1000) do i
                sim = simulate(policy; s)
                outcome = sim.bs[end].focused
                push!(sims, (;outcome, sim.cs, v0, vd))
            end
        end
    end
    Table(sims)
end

function run_sims(make_policy, name, G)
    println("Simulating $name policy on $(length(G)) MDPs")
    mkpath("tmp/sims/$name/")
    X = @showprogress pmap(G) do g
        m = make_mdp(g)
        cache("tmp/sims/$name/" * id(m)) do
            policy = make_policy(m)
            generate_sims(policy)
        end
    end
    flat = mapmany(G, X.data) do prm, sim
        map(sim) do t
            rt = length(t.cs)
            fix_prop = counts(t.cs, 1:2)[2] / rt
            (;prm..., t.outcome, t.v0, t.vd, fix_prop, rt,
             first_fix=t.cs[1], last_fix=t.cs[end])
        end
    end
    flat |> Vector{typeof(flat[1])} |> Table |> CSV.write("results/sim_$name.csv")
end

run_sims(RandomPolicy, "random", G(switch_cost=0, miss_cost=0))

run_sims("optimal", G) do m
    V = cache("tmp/V/" * id(m)) do
        println("Computing value function")
        V = ValueFunction(m)
        V(initial_belief(m))
        V
    end
    OptimalPolicy(V)
end
