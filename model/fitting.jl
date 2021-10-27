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
# %% --------
function load_data()
    simple = mapreduce(vcat, VERSIONS) do version
        CSV.read("../data/$version/simple-recall.csv", DataFrame)
    end
    strengths = @chain simple begin
        @subset :block .== maximum(:block)
        @subset .! :practice
        @transform @byrow :strength = 5 * (:response_type == "correct") - log(:rt)
        @by [:wid, :word] :strength = mean(:strength)
    end

    multi = mapreduce(vcat, VERSIONS) do version
        CSV.read("../data/$version/multi-recall.csv", DataFrame)
    end
    wid_counts = countmap(multi.wid)
    filter!(multi) do row
        !row.practice && wid_counts[row.wid] == 20
    end
    multi.presentation_times = map(Vector{Float64} ∘ JSON.parse, multi.presentation_times)

    @chain multi begin
        @subset @byrow !ismissing(:first_word) && !:practice
        leftjoin(strengths, on=[:wid, :first_word=>:word], renamecols = ""=>"_first")
        leftjoin(strengths, on=[:wid, :second_word=>:word], renamecols = ""=>"_second")
        @transform :choose_first = :first_seen .== :choice
        @select :wid :presentation_times :choose_first :response_type :strength_first :strength_second :choice_rt
    end
end
df = load_data()

# %% --------
@everywhere begin
    qq = .1:.2:.9
    function compute_metrics(rt, nfix)
        (q_rt = quantile(rt, qq), μ_rt = mean(rt), σ_rt = std(rt),
         p_nfix = counts(nfix, 1:5) ./ length(nfix),
         μ_nfix = mean(nfix), σ_nfix = std(nfix))
    end
    missing_metrics = map(x->missing, compute_metrics([1.], [1]))
end

target = @chain df begin
    @subset :response_type .== "correct"
    @with compute_metrics(:choice_rt, length.(:presentation_times))
end

# %% --------
prms = grid(
    α = 1:5,
    β = 1:10,
    threshold = 10:10:70,
    switch_cost = 0:.5:5
)

size(prms)
# %% --------
mkpath("tmp/try_one")

@everywhere function try_one(prm)
    name = replace(string(prm),  r"[() ]" => "")
    cache("tmp/try_one/$name") do
        # WARNING: assuming 12 second max
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

        isempty(sims) && return (;error_rate, missing_metrics)

        rt = map(sims) do sim
            #100 .* length(sim.cs)
            -100 * sim.total_cost  # include switch cost in RT
        end
        nfix = map(sims) do sim
            sum(diff(sim.cs) .!= 0) + 1
        end
        (;error_rate, compute_metrics(rt, nfix)...)
    end
end

sumstats = @showprogress pmap(try_one, prms)

serialize("results/sumstats", sumstats)

# %% ==================== Analyze ====================

sumstats = deserialize("results/sumstats")

err_rt = map(sumstats) do s
    sum(abs.(s.q_rt .- target.q_rt))
end
err_nfix = map(sumstats) do s
    sum(abs.(s.p_nfix .- target.p_nfix))
end
prms[argmin(err_rt)]

# %% ==================== RT ====================

function remove_bad!(X, sds=2)
    bad = minimum(X) + sds * std(X[:])
    X[X .> bad] .= NaN
    X
end

figure() do
    sminimum(err_rt, :threshold, :switch_cost)  |> remove_bad! |> heatmap
end
# %% --------
figure() do
    sminimum(err_rt, :α, :β) |> remove_bad! |> heatmap
end

# %% ==================== N fix ====================

figure() do
    sminimum(err_nfix, :threshold, :switch_cost) |> remove_bad! |> heatmap
end

figure() do
    sminimum(err_nfix, :α, :β) |> remove_bad! |> heatmap
end

# %% ====================  ====================

figure() do
    smaximum(rt, :threshold, :switch_cost) |> heatmap
end

figure() do
    smaximum(nfix, :α, :β) |> heatmap
end

figure() do
    smaximum(getfield.(sumstats, :μ_nfix), :α, :β) |> heatmap
end

bad_err = getfield.(sumstats, :error_rate) .> .15
loss = err_rt .+ 500err_nfix .+ 1e10bad_err
rank = sortperm(loss[:])



# %% ==================== Prop fix ====================

df






