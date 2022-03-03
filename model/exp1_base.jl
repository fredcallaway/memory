@everywhere include("common.jl")
@everywhere include("exp1_simulate.jl")
mkpath("results/exp1")

pretest = load_data("exp1/pretest")
trials = load_data("exp1/trials")

@everywhere trials = $trials
@everywhere pretest = $pretest

N_SOBOL = 100_000

function compute_sumstats(name, make_policies, prms; read_only = true)
    mkpath("cache/exp1_$(name)_sumstats")
    map = read_only ? asyncmap : pmap
    @showprogress map(prms) do prm
        cache("cache/exp1_$(name)_sumstats/$(stringify(prm))"; read_only) do
            exp1_sumstats(simulate_exp1(make_policies, prm))
        end
    end;
end

# %% ==================== summary statistics ====================

@everywhere function unroll_trial!(P, rt, response_type; dt)
    max_step = size(P, 1)
    n_step = round(Int, rt / dt)
    P[1:n_step, 1] .+= 1 
    outcome = response_type == "correct" ? 2 : 3
    P[n_step+1:max_step, outcome] .+= 1
end

@everywhere function unroll_time(trials; dt=ms_per_sample, maxt=15000)
    @chain trials begin
        groupby(:pretest_accuracy)
        combine(_) do d
            P = zeros(Int(maxt/dt), 3)
            for t in eachrow(d)
                unroll_trial!(P, t.rt, t.response_type; dt)
            end
            P ./= nrow(d)
            Ref(P)  # prevents unrolling the array
        end
        @orderby :pretest_accuracy
        @with combinedims(:x1)
        KeyedArray( 
            time=dt:dt:maxt, 
            event=[:thinking, :recalled, :skipped], 
            pretest_accuracy=0:0.5:1
        )
    end 
end

@everywhere function exp1_sumstats(trials)
    try
        rt = @chain trials begin
            groupby([:response_type, :pretest_accuracy, :judgement])
            @combine begin
                :μ = mean(:rt)
                :σ = std(:rt)
                :n = length(:rt)
            end
        end
        (;rt, unrolled = unroll_time(trials))
    catch
        missing
    end
end

target = exp1_sumstats(trials)

# %% ==================== loss function ====================

response_rate(x) = x.acc_n(response_type="correct") ./ ssum(x.acc_n, :response_type)
pretest_dist(x) = normalize(ssum(x.acc_n, :response_type))

function marginal_loss(pred)
    size(pred.acc_rt) == size(target.acc_rt) || return Inf
    sum(squared.(response_rate(pred) .- response_rate(target))) +
    sum(squared.(pretest_dist(pred) .- pretest_dist(target))) +
    mean(squared.((target.rt_μ .- pred.rt_μ) ./ target.rt_σ))
end

function acc_rt_loss(pred)
    size(pred.acc_rt) == size(target.acc_rt) || return Inf
    dif = (target.acc_rt .- pred.acc_rt)[:]
    mask = (!ismissing).(target.acc_rt[:])
    l = mean(squared.(dif[mask]) / 1000^2)
    ismissing(l) ? Inf : l
end

function judge_rt_loss(pred)
    size(pred.judge_rt) == size(target.judge_rt) || return Inf
    dif = (target.judge_rt .- pred.judge_rt)[:]
    mask = (!ismissing).(target.judge_rt[:])
    l = mean(squared.(dif[mask]) / 1000^2)
    ismissing(l) ? Inf : l
end

function cum_prob_loss(x)    
    mean(squared.(target.p_correct .- x.p_correct)) +
    mean(squared.(target.p_skip .- x.p_skip))
end

function acc_judge_loss(x)
    acc_rt_loss(x) + judge_rt_loss(x)
end

function loss(ss)
    mae(target.unrolled(time = <(10000)), ss.unrolled(time = <(10000)))
end
