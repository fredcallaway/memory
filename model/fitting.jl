using ProgressMeter
using JSON
@everywhere begin
    using CSV, DataFrames, DataFramesMeta
    using StatsBase
    include("utils.jl")
    include("figure.jl")
    include("constants.jl")
    include("mdp.jl")
    include("optimal_policy.jl")
end
# time out proportion
# average fixation duration
# average fixation length

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)

# %% ==================== Simulate ====================

@everywhere begin
    qq = .1:.2:.9
    function compute_metrics(rt, nfix)
        (q_rt = quantile(rt, qq), μ_rt = mean(rt), σ_rt = std(rt),
         p_nfix = counts(nfix, 1:5) ./ length(nfix),
         μ_nfix = mean(nfix), σ_nfix = std(nfix))
    end
    missing_metrics = map(x->missing, compute_metrics([1.], [1]))

    make_mdp(prm) = MetaMDP(;prm.step_size, prm.max_step, prm.threshold, prm.switch_cost, prior=(prm.α, prm.β))
end

prms = grid(
    step_size=[4],
    max_step=[120],
    α = 1:5,
    β = 1:10,
    threshold = 20:5:60,
    switch_cost = 0:.25:2
)

size(prms)
mkpath("tmp/try_one")

@everywhere function try_one(prm; N=50000, disable_cache=false)
    name = replace(string(prm),  r"[() ]" => "")
    cache("tmp/try_one/$name"; disable=disable_cache) do
        # WARNING: assuming 12 second max
        pol = OptimalPolicy(make_mdp(prm))
        sims = repeatedly(N) do
            sim = simulate(pol)            
        end;
        filter!(sims) do sim
            sim.b.focused != -1
        end
        error_rate = 1 - (length(sims) / N)
        isempty(sims) && return (;error_rate, missing_metrics)

        rt = map(sims) do sim
            # 100 * sim.total_cost  # include switch cost in RT
            100 * (sim.total_cost - sim.fix_log.n * pol.m.switch_cost)
        end
        nfix = map(sims) do sim
            sim.fix_log.n
        end
        (;error_rate, compute_metrics(rt, nfix)...)
    end
end

sumstats = @showprogress pmap(try_one, prms)
serialize("results/sumstats", sumstats)

# %% ==================== Human ====================

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
df = @chain load_data() begin
    @rtransform(
        :duration_first = sum(:presentation_times[1:2:end]),
        :duration_second = sum(:presentation_times[2:2:end])
    )
end

target = @chain df begin
    @subset :response_type .== "correct"
    @with compute_metrics(:choice_rt, length.(:presentation_times))
end

# %% ==================== Analyze ====================
sumstats = deserialize("results/sumstats")
# %% --------

# rt_loss(s) = abs(s.μ_rt - target.μ_rt) / target.σ_rt
# nfix_loss(s) = abs(s.μ_nfix - target.μ_nfix) / target.σ_nfix
rt_loss(s) = sum(abs.(s.q_rt .- target.q_rt))
nfix_loss(s) = sum(abs.(s.p_nfix .- target.p_nfix))
err_loss(s) = (s.error_rate > .15)
full_loss(s) = rt_loss(s) + 2000 * nfix_loss(s) + 1e10 * err_loss(s)

# %% --------

loss = full_loss.(sumstats)
best = keymin(loss)
try_one(best)

# %% ==================== RT ====================

function remove_bad!(X, sds=2)
    bad = minimum(X) + sds * std(X[:])
    X[X .> bad] .= NaN
    X
end

S = sumstats(step_size=4, max_step=120)
X = rt_loss.(S)

figure() do
    X(;best.threshold, best.switch_cost) |> heatmap
end

figure() do
    X(;best.α, best.β) |> heatmap
end

figure() do
    sminimum(X, :threshold, :switch_cost)  |> remove_bad! |> heatmap
end

figure() do
    sminimum(X, :α, :β) |> remove_bad! |> heatmap
end

# %% ==================== N fix ====================
X = nfix_loss.(S)

figure() do
    X(;best.threshold, best.switch_cost) |> heatmap
end

figure() do
    X(;best.α, best.β) |> heatmap
end

figure() do
    sminimum(X, :threshold, :switch_cost)  |> remove_bad! |> heatmap
end

figure() do
    sminimum(X, :α, :β) |> remove_bad! |> heatmap
end

# %% ==================== Write simulation ====================

@everywhere function make_frame(pol, N=10000; ms_per_sample=100)
    sims = map(1:N) do i
        sim = simulate(pol; fix_log=FullFixLog())
        strength_first, strength_second = sim.s
        presentation_times = sim.fix_log.fixations .* ms_per_sample #.+ pol.m.switch_cost * ms_per_sample
        outcome = sim.b.focused

        (;strength_first, strength_second, presentation_times, outcome,
         duration_first = sum(presentation_times[1:2:end]),
         duration_second = sum(presentation_times[2:2:end]))
    end
    DataFrame(sims[:])
end

pol = OptimalPolicy(make_mdp(best))
try_one(best).μ_rt
target.μ_rt
sim = make_frame(pol)
sim 
@chain sim begin
    @rsubset :outcome != -1
    @with mean(:duration_first .+ :duration_second)
end
sim |> CSV.write("results/sim_new_optimal.csv")



