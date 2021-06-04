using Revise
includet("binomial_accumulator.jl")
include("figure.jl")
includet("utils.jl")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)

# %% --------
m = MetaMDP(step_size=4, max_step=60, threshold=100, sample_cost=1, switch_cost=3, miss_cost=100, allow_stop=true)
V = ValueFunction(m)
@time V(initial_belief(m))
pol = SoftOptimalPolicy(V; β=10.)

# %% --------

function fix_matrix(pol, s; N=10000)
    F = mapreduce(hcat, 1:N) do i
        sim = simulate(pol; s)
        [sim.cs; fill(0, m.max_step - length(sim.cs))]
    end
end

function plot_fix(F; kws...)
    plot(;xlabel="Timestep", ylabel="Probability", kws...)
    plot!([mean(F .== 0; dims=2)], color=:lightgray, label="Already recalled")
    plot!([mean(F .== 1; dims=2) mean(F .== 2; dims=2)], color=[1 2], 
        label=["Attend more memorable" "Attend less memorable"],
        legend=:topleft)
end

# %% --------

states = [
    (0.7, 0.6),
    (0.7, 0.5),
    (0.8, 0.7),
    (0.8, 0.6),
]
F_opts = map(states) do s
    fix_matrix(pol, s) 
end

figure("fixation_by_strength") do
    plots = map(F_opts, states) do F, s
        plot_fix(F, title=string(s))
    end
    plot(plots..., size=(600,450), legend=false)
end

# %% --------
#sim = simulate(pol; s)
#pol.m.allow_stop


figure("simple_fixation2") do
    F = fix_matrix(mutate(pol; β=5.), (0.3, 0.2); N=1000)
    plot(;xlabel="Timestep", ylabel="Probability", ylim=(-0.02, 1.22))
    plot!([mean(F .== 0; dims=2)], color=:lightgray, label="Already recalled")
    plot!([mean(F .== 1; dims=2) mean(F .== 2; dims=2)], color=[2 1], 
        label=["Attend more memorable" "Attend less memorable"],
        legend=:topleft)
end

# %% --------
s = (0.5, 0.5)
F_noswitch = fix_matrix(RandomSwitchPolicy(m, 0), s)

figure("optimal_vs_none") do
    plot([mean(F_noswitch .== 0; dims=2)], color=:gray, label="No Switch")
    plot!([mean(F_opt .== 0; dims=2)], color=4, label="Optimal")
    s1, s2 = s
    plot!(legend=:topleft, title="$s1 vs. $s2", ylabel="P(Remembered)", xlabel="Timesteps")
end

# %% --------

noswitch = RandomSwitchPolicy(m, 0)
simulate(pol).total_cost
monte_carlo(1000) do
    simulate(noswitch).total_cost
end
monte_carlo(1000) do
    simulate(pol).total_cost
end

m1 = MetaMDP{1}(step_size=4, max_step=80, threshold=100, sample_cost=1, miss_cost=0)
monte_carlo(10000) do
    simulate(RandomSwitchPolicy(m1, 0)).total_cost
end


# %% --------
figure("random_fix") do
    plot([mean(F .== 0; dims=2)], color=:lightgray, label="Remembered")
    s1, s2 = s
    plot!([mean(F .== 1; dims=2) mean(F .== 2; dims=2)], color=[1 2], label=["Fixate $s1" "Fixate $s2"])
    plot!(legend=:topleft, title="$s1 vs. $s2", ylabel="Probability", xlabel="Timesteps")
    # plot!(F[:, 1:10] .- 1, color=:black, alpha=0.1)
end



# %% --------
# using TypedTables
# using SplitApplyCombine

# sims = map(1:1000) do i
#     simulate(pol; s)
# end |> Table
# # %% --------
# counts(sims[1].cs)
# p_left = map(sims) do t
#     counts(t.cs, 1:2)[1] / length(t.cs)
# end

# mean(p_left)

# s