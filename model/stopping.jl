@everywhere include("mdp.jl")
@everywhere include("utils.jl")
include("figure.jl")
include("optimal_policy.jl")

using ProgressMeter
using CSV, DataFrames, DataFramesMeta

# %% ==================== Setup ====================

m = MetaMDP{1}(allow_stop=true, step_size=4, max_step=120, threshold=40, 
               sample_cost=1, miss_cost=50, prior=(1, 1))
# %% --------

m = MetaMDP{1}(allow_stop=true, step_size=1, max_step=400, threshold=100, 
               sample_cost=2, miss_cost=400, prior=(1, 1))


B = BackwardsInduction(m)

X = map(Iterators.product(0:m.max_step-1, 0:m.threshold)) do (stp, heads)
    stp == m.max_step && return 0.
    tails = stp * m.step_size - heads
    tails < 0 && return NaN
    b = Belief{1}(stp, 1, [heads], [tails])
    value(B, b)
end

figure() do
    heatmap(X', 
        c=:Blues, 
        # clim=(-60, 0),
        # c=:RdBu_11, clim=(-100, 0),
        #xaxis=("Time", 0:25:100),
        #yaxis=("Evidence", -5:10:50, 0:10:50))
        xlab="Time",
        ylab="Recall Progress",
        xlim=(0,100),
        # ylim=(0.5,52.1),
        # xticks=(1:50:201, string.(0:50:200)),
        # yticks=(1:10:51, string.(0:10:50))
    )
    # hline!([51], color=:black)
end

# %% ==================== Strength distribution ====================
m = mutate(m, step_size=4, miss_cost=3, sample_cost=.01, prior=(1,1), threshold=80)
pol = SoftOptimalPolicy(m, 3)
rt = map(1:10000) do i
    simulate(pol; fix_log=RTLog()).fix_log.rt
end
figure() do
   histogram(rt)
end

# %% --------

m = MetaMDP{1}(allow_stop=true, step_size=4, max_step=120,
               sample_cost=.012, miss_cost=3, prior=(1,2), threshold=80)
pol = SoftOptimalPolicy(m, 10)

function sample_states(N=10000)
    states = [Float64[] for i in 1:3]
    for i in 1:N
        s = sample_state(m)
        n_correct = (simulate(pol; s).b.focused == 1) + (simulate(pol; s).b.focused == 1)
        push!(states[n_correct + 1], s[1])
    end
    states
end

states = sample_states()

figure() do
    map(states) do x
        histogram!(x, alpha=0.5, bins=0:.05:1)
    end
end
# %% --------

sims = mapreduce(vcat, states, [0, 0.5, 1]) do ss, pre_correct
    map(ss) do strength
        strength = logistic(logit(strength) + randn())
        sim = simulate(pol; s = (strength,), fix_log=RTLog())
        (;pre_correct, strength=sim.s[1], outcome=sim.b.focused, rt=sim.fix_log.rt * 100)
    end
end

df = DataFrame(sims)

figure() do
    @chain df begin
        @by([:outcome, :pre_correct], :rt = mean(:rt))
        @orderby(:outcome, :pre_correct)
        @df scatter!(:pre_correct, :rt, group=:outcome, smooth=true)
    end
end

# df |> CSV.write("results/stopping_sim2.csv")

# %% ==================== Simulate ====================

function make_frame(pol, N=10000; ms_per_sample=100)
    sims = map(1:N) do i
        sim = simulate(pol; fix_log=RTLog())
        (;strength=sim.s[1], outcome=sim.b.focused, rt=sim.fix_log.rt * ms_per_sample)
    end
    DataFrame(sims[:])
end

function plot_rts(m)
    df = SoftOptimalPolicy(m, 30) |> make_frame
    figure(title=mean(df.outcome .== 1)) do
        @chain df begin
            # @rtransform(:strength=log(:strength))
            @rtransform(:strength=round(:strength .+ .1randn(), digits=1) + .001)
            @by([:outcome, :strength], :rt = mean(:rt))
            @orderby(:outcome, :strength)
            @df scatter!(:strength, :rt, group=:outcome, smooth=true)
        end
    end
end

m = MetaMDP{1}(allow_stop=true, step_size=4, max_step=120,
               sample_cost=.012, miss_cost=3, prior=(1,2), threshold=80)

plot_rts(m)
# %% --------


# df |> CSV.write("results/stopping_sim.csv")

# %% --------
df = DataFrame(x=[:one, :two], y = [1, 2], z=["null", "null"])
unstack(df, :x, :y)



 