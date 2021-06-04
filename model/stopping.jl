@everywhere include("binomial_accumulator.jl")
@everywhere include("utils.jl")
using ProgressMeter

# %% ==================== Simulation ====================


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

# %% ==================== Policy heatmap ====================
#include("figure.jl")
max_step = 200
m = MetaMDP{1}(;step_size=1, max_step, threshold=50, sample_cost=1, miss_cost=100, allow_stop=true)
V = ValueFunction(m); @time println(V(initial_belief(m)))
pol = OptimalPolicy(V)

X = map(Iterators.product(0:max_step, 0:49)) do (t, h)
    t == m.max_step && return 0.
    h > t && return NaN
    b = Belief{1}(t, 1, ((1+h, 1+t-h),))
    float(act(pol, b))
end

B = map(Iterators.product(0:max_step, 0:50)) do (t, h)
    Belief{1}(t, 1, ((1+h, 1+t-h),))
end

figure() do
    heatmap(X', c=:viridis, clim=(0, 1.5), cbar=false,
        #xaxis=("Time", 0:25:100),
        #yaxis=("Evidence", -5:10:50, 0:10:50))
        xlab="Time",
        ylab="Evidence",
        ylim=(0.5,52.1),
        xticks=(1:50:201, string.(0:50:200)),
        yticks=(1:10:51, string.(0:10:50))
    )
    hline!([51], color=:black)
end

# %% ==================== Threshold ====================

X



# %% --------
#t = 99
#h = 48
#b = Belief{1}(t, 1, ((1+h, 1+t-h),))
b = B[100, 50]

#X[100, 49]
#act(pol, b)
# %% --------
size(X)
p, b1, = results(m, b, 1)[2]
results(m, b1, 1)
b1
b1
Q(V, b, 0)
