using Revise
includet("binomial_accumulator.jl")
include("figure.jl")
includet("utils.jl")
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)

m = MetaMDP(step_size=4, max_step=60, threshold=100, sample_cost=1, miss_cost=0)
# %% --------
V = ValueFunction(m)
@time V(initial_belief(m))

# %% --------

function fix_matrix(pol, s; N=10000)
    F = mapreduce(hcat, 1:10000) do i
        sim = simulate(pol; s)
        [sim.cs; fill(0, m.max_step - length(sim.cs))]
    end
end

function plot_fix(F; kws...)
    plot(;xlabel="Timestep", ylabel="P(fixate)", kws...)
    plot!([mean(F .== 0; dims=2)], color=:lightgray)
    plot!([mean(F .== 1; dims=2) mean(F .== 2; dims=2)], color=[1 2], label=["More memorable" "less memorable"])
end



# %% --------
pol = SoftOptimalPolicy(V; β=.3)

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
        plot_fix(F, legend=false, title=string(s))
    end
    plot(plots..., size=(600,450))
end

# %% --------

F_noswitch = fix_matrix(RandomSwitchPolicy(m, 0), s)

figure("optimal_vs_none") do
    plot([mean(F_noswitch .== 0; dims=2)], color=:gray, label="No Switch")
    plot!([mean(F_opt .== 0; dims=2)], color=4, label="Optimal")
    s1, s2 = s
    plot!(legend=:topleft, title="$s1 vs. $s2", ylabel="P(Remembered)", xlabel="Timesteps")
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