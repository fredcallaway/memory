using ProgressMeter
using CSV, DataFrames, DataFramesMeta
using JSON
@everywhere begin
    include("binomial_accumulator.jl")
    include("utils.jl")
    include("figure.jl")
    include("constants.jl")
    include("backwards_induction.jl")
end
# time out proportion
# average fixation duration
# average fixation length

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)

function load_multi()
    multi = mapreduce(vcat, VERSIONS) do version
        CSV.read("../data/$version/multi-recall.csv", DataFrame)
    end
    wid_counts = countmap(multi.wid)
    filter!(multi) do row
        !row.practice && wid_counts[row.wid] == 20
    end
    multi.presentation_times = map(Vector{Float64} ∘ JSON.parse, multi.presentation_times)
    multi
end

df = @chain load_multi() begin
    @subset @byrow :response_type == "correct" && !ismissing(:first_word)
end

# %% --------
@everywhere qq = .1:.2:.9

@everywhere function compute_metrics(rt, nfix)
    (q_rt = quantile(rt, qq), μ_rt = mean(rt), σ_rt = std(rt),
     q_nfix = quantile(nfix, qq), μ_nfix = mean(nfix), σ_nfix = std(nfix))
end

target = compute_metrics(skipmissing(df.typing_rt), skipmissing(length.(df.presentation_times)))

# %% --------
prms = grid(
    α = 1:5,
    β = 1:5,
    threshold = 10:10:50,
    switch_cost = 0:5
)

size(prms)
# %% --------

@everywhere function try_one(prm)
    m = MetaMDP(;step_size=4, max_step=120, prm.threshold, sample_cost=1, prm.switch_cost, miss_cost=0, prior=(prm.α, prm.β))
    pol = BIPolicy(m)
    sims = repeatedly(10000) do
        simulate(pol)
    end
    error_rate = length(sims) \ mapreduce(+, sims) do sim
        sim.bs[end].focused == -1
    end
    filter!(sims) do sim
        sim.bs[end].focused != 1
    end
    nfix = map(sims) do sim
        sum(diff(sim.cs) .!= 0) + 1
    end
    rt = map(sims) do sim
        100 .* length(sim.cs)
    end
    (;q_rt = quantile(rt, qq), μ_rt = mean(rt), σ_rt = std(rt),
      q_nfix = quantile(nfix, qq), μ_nfix = mean(nfix), σ_nfix = std(nfix),
      error_rate)
end

sumstats = @showprogress pmap(try_one, prms)

# %% --------
err_rt = map(sumstats) do s
    sum(abs.(s.q_rt .- target.q_rt))
end
err_nfix = map(sumstats) do s
    sum(abs.(s.q_nfix .- target.q_nfix))
end
prms[argmin(err_rt)]

# %% --------

figure() do
    smaximum(rt, :threshold, :switch_cost) |> heatmap
end

figure() do
    smaximum(nfix, :α, :β) |> heatmap
end

figure() do
    smaximum(getfield.(sumstats, :μ_nfix), :α, :β) |> heatmap
end

# %% ==================== RT ====================

bad_err = getfield.(sumstats, :error_rate) .> .15
loss = err_rt .+ 500err_nfix .+ 1e10bad_err
rank = sortperm(loss[:])

prms[rank[1:10]]
getfield.(sumstats, :q_rt)[rank[7]]
target.q_rt

prm = prms[rank[1]]

err_nfix[rank[3]]










