include("utils.jl")
include("mdp.jl")
include("optimal_policy.jl")
include("box.jl")

if isinteractive()
    Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2]-2)
end


using Optim
using ProgressMeter
using JuliennedArrays

const MAX_TIME = 15000
const MS_PER_SAMPLE = 100
const MAX_STEP = Int(MAX_TIME / MS_PER_SAMPLE)

SMOOTHING = 1e-5

quantize(x, q=MS_PER_SAMPLE) = q * cld(x, q)
load_data(name) = CSV.read("../data/processed/$name", DataFrame, missingstring="NA")
stringify(nt::NamedTuple) = replace(string(map(x->round(x; digits=8), nt::NamedTuple)), ([" ", "(", ")"] .=> "")...)

using GLM
macro regress(data, expr)
    :(lm(@formula($expr), $(esc(data))))
end
get_coef(m) = coef(m)[2], confint(m)[2, :]

function compute_cached(job_fn, job_name, prms; read_only=false, enable_cache=true, overwrite=false, catch_errors=true)
    dir = "cache/$(RUN)_$(EXP_NAME)_$(job_name)"
    mkpath(dir)
    map = read_only ? asyncmap : pmap
    @showprogress "$job_name "  map(prms) do prm
        cache("$dir/$(hash(prm))"; read_only, disable=!enable_cache, overwrite) do
            try
                job_fn(prm)
            catch
                if !catch_errors
                    rethrow()
                else
                    missing
                end
            end
        end
    end;
end

function sample_strengths(pol, N=10000; between_σ, within_σ)
    @assert pol.m.prior.σ^2 ≈ between_σ^2 + within_σ^2
    underlying_strength_dist = Normal(pol.m.prior.μ, between_σ)
    trial_strength_shift = Normal(0, within_σ)

    map(1:N) do i
        underlying_strength = rand(underlying_strength_dist)
        pretest_accuracy = 2 \ mapreduce(+, 1:2) do i
            s = (underlying_strength + rand(trial_strength_shift),)
            simulate(pol; s).b.focused == 1
        end
        strength = underlying_strength + rand(trial_strength_shift)
        (strength, pretest_accuracy)
    end
end

function pretest_mdp(prm)
    time_cost = @isdefined(PRETEST_COST) ? (MS_PER_SAMPLE / MAX_TIME) * .25 : 0
    MetaMDP{1}(;allow_stop=true, max_step=MAX_STEP, miss_cost=1,
        prm.threshold, prm.noise, sample_cost=prm.sample_cost + time_cost,
        prior=Normal(prm.drift_μ, prm.drift_σ),
    )
end

function empirical_distribution(x)
    fit(DiscreteNonParametric, max.(1, round.(Int, x ./ MS_PER_SAMPLE)))
end

# %% ==================== likelihood stuff ====================

function reparameterize(prm)
    drift_σ = √(prm.between_σ^2 + prm.within_σ^2)
    judgement_noise = 0.5 .* drift_σ
    prm = (;prm..., drift_σ, judgement_noise)

    if hasfield(typeof(prm), :αθ_stop)
        prm = (;prm..., θ_stop = prm.αθ_stop / prm.α_stop)
    end
    if hasfield(typeof(prm), :αθ_switch)
        prm = (;prm..., θ_switch = prm.αθ_switch / prm.α_switch)
    end
    if hasfield(typeof(prm), :αθ_ndt)
        prm = (;prm..., θ_ndt = prm.αθ_ndt / prm.α_ndt)
    end
    prm
end

sample_params(box, N) = map(reparameterize, sobol(N, box))

function smooth_uniform!(x, ε::Float64=1e-6)
    x .*= (1 - ε)
    x .+= (ε / length(x))
    x
end

# convolves two binned pdfs over a non-negative domain
function convolve!(result::AbstractVector, pdf_x::AbstractVector, pdf_y::AbstractVector)
    # note: we identify bins with the left bound, so pdf_y[1] is p(Y = 0)
    for z in axes(pdf_x, 1)
        result[z] = sum(1:z) do x
            y = z - x
            @inbounds pdf_x[x] * pdf_y[y + 1]
        end
    end
end

function convolve!(result::AbstractArray, pdf_x::AbstractArray, pdf_y::AbstractVector)
    # we always convolve along the first dimension (RT / duration)
    for (rslice, xslice) in zip(Slices(result, 1), Slices(pdf_x, 1))
        convolve!(rslice, xslice, pdf_y)
    end
end

function convolve!(result, pdf_x::KeyedArray, y::Distribution)
    pdf_y = diff([0; cdf(y, axiskeys(pdf_x, 1))])
    convolve!(result, pdf_x, pdf_y)
end

function likelihood(model, human, ndt, tmp=zeros(size(model)); ε::Float64=SMOOTHING)
    convolve!(tmp, model, ndt)
    smooth_uniform!(tmp, ε)
    crossentropy(human, tmp)
end

function optimize_ndt(model::KeyedArray, human::KeyedArray; ε::Float64=SMOOTHING)
    tmp = zeros(size(model))
    optimize([10., 10.]) do x
        any(xi < 0 for xi in x) && return Inf
        likelihood(model, human, Gamma(x...), tmp; ε)
    end
end

function skip_rt_hist(trials; dt=MS_PER_SAMPLE, maxt=MAX_TIME)
    @chain trials begin
        @rsubset :response_type == "empty" && :rt < MAX_TIME
        @rtransform :rt = quantize(:rt, dt)
        wrap_counts(rt = dt:dt:maxt)
        normalize!
    end
end

function optimize_stopping_model(human, α_ndt, θ_ndt; ε::Float64=SMOOTHING)
    ndt_dist = Gamma(α_ndt, θ_ndt)
    ndt_only = similar(human)    
    ndt_only .= diff([0; cdf(ndt_dist, axiskeys(human, 1))])

    # RT = NDT + stop_time; NDT is fixed, so we can optimize the stopping dist
    # in the same way we optimize NDT.
    α, θ = optimize_ndt(ndt_only, human; ε).minimizer
    Gamma(α, θ / MS_PER_SAMPLE)  # convert to units of samples
end

function load_fit(name, results=RESULTS)
    top = deserialize("$results/fits/$name/top")
    first(eachrow(top))
end


function write_tex(name, x)
    fname = "$RESULTS/tex/$name.tex"
    mkpath(dirname(fname))
    str = string(x) * "\\unskip"
    println(fname, ":\n", x)
    open(fname, "w") do f
        write(f, str)
    end
    return
end

write_tex(name) = x -> write_tex(name, x)

function lower_ci(ef, effect)
    ci = getfield(ef, effect)[2]
    fillnan(ci[1])
end

function fmt_ci((est, (lo, hi)), digits=3; convert_to_seconds=true, negate=false)
    est, lo, hi = map((est, lo, hi)) do x
        if convert_to_seconds
            x /= 1000
        end
        if negate
            x *= -1
        end
        x
    end
    if negate
        lo, hi = hi, lo
    end
    est, lo, hi = fmt.(digits, (est, lo, hi))
    "\$B = $est\$; 95\\% CI [$lo, $hi]"
end
