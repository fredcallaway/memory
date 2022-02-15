include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("figure.jl")
using DataFrames, DataFramesMeta, CSV

# %% ==================== Likelihood ====================

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.1, 
    threshold=8, noise=2, max_step=60, prior=Normal(0.5, 2)
)

trials = CSV.read("../data/processed/stopping.csv", DataFrame)


# %% ==================== Simulate ====================
# explicit cost is .1 per second

m = MetaMDP{1}(allow_stop=true, miss_cost=3, sample_cost=.03, 
    threshold=14, noise=2, max_step=150, prior=Normal(0, 1)
)



function sample_states(pol, N=10000)
    states = [Float64[] for i in 1:3]
    for i in 1:N
        s = sample_state(pol.m)
        n_correct = (simulate(pol; s).b.focused == 1) + (simulate(pol; s).b.focused == 1)
        push!(states[n_correct + 1], s[1])
    end
    states
end

function make_frame(pol, N=10000)
    states = sample_states(pol, N)
    # rt_noise=Gamma(10, 30)
    sims = mapreduce(vcat, states, [0, 0.5, 1]) do ss, pre_correct
        map(ss) do strength
            strength = strength + rand(Normal(0, .5))
            # strength = logistic(logit(strength) + randn())
            sim = simulate(pol; s = (strength,), fix_log=RTLog())
            post = posterior(pol.m, sim.b)[1]
            (;pre_correct, strength, outcome=sim.b.focused, rt=sim.fix_log.rt * 100,
              μ_post=post.μ, σ_post=post.σ)
        end
    end
    DataFrame(sims[:])
    # @rtransform(DataFrame(sims[:]),
    #     :judgement = rand(Normal(:μ_post, :σ_post)),
    #     :rt = :rt + rand(rt_noise)
    # )
end

opt_pol = OptimalPolicy(m)
make_frame(opt_pol) |> CSV.write("results/stopping_sim.csv")

# %% --------

rt = @subset(trials, :response_type .== "empty").rt
rand_pol = StopDistributionPolicy2(m, fit(Gamma, rt./100))

make_frame(rand_pol) |> CSV.write("results/stopping_sim_rand.csv")




