include("figure.jl")
include("utils.jl")
using ProgressMeter
using Distributions

#= 
Question: how many samples from a binomial does it take to reach a threshold
Conclusion: the expected RT is threshold / (p * n) ... duh!
=#

# %% --------

function run_sim(n, p, thresh)
    d = Binomial(n, p)
    x = 0
    i = 0
    while x < thresh
        i += 1
        x += rand(d)
    end
    return i
end

function run_many(n, p, thresh; N=1000)
    monte_carlo(1000) do
        run_sim(n, p, thresh)
    end
end

# %% --------
function make_hist(n, p, thresh; N=10000, maxn=ceil(Int, 4*thresh/(n*p)))
    x = zeros(Int, maxn)
    for i in 1:N
        x[run_sim(n, p, thresh)] += 1
    end
    x
end

figure() do
    bar(make_hist(1, 0.4, 10))
end

# %% --------

n = 5
p = 0.2
thresh = 20
x = [run_sim(n, p, thresh) for i in 1:100000]
figure() do
    histogram(x, normalize=:pdf, bins=0.5:maximum(x)+0.5)
    start = (thresh/n)
    d = fit(Gamma, x .- start)
    plot!(start:maximum(x), pdf.(d, 0:maximum(x)-start))
    d = 1:200
    plot!(d./n, n .* pdf(NegativeBinomial(thresh, p), d .- thresh))
end



# %% ==================== Moment matching for NegativeBinomial and Gamma ====================

r, p = 10,  0.1
μ, σ2 = juxt(mean, var)(NegativeBinomial(r, p))

@assert r ≈ μ^2 / (σ2 - μ)
@assert p ≈ μ / σ2
@assert μ ≈ ((1 - p) * r) / p
@assert σ2 ≈ ((1-p) * r) / p^2

α = μ^2 / σ2
θ = σ2 / μ

gamma = Gamma(α, θ)
negbin = NegativeBinomial(r, p)

@assert mean(gamma) ≈ mean(negbin)
@assert var(gamma) ≈ var(negbin)

figure() do
    plot!(negbin; markeralpha=0)
    plot!(Gamma(α, θ), ls=:dash)
end

function rp_to_αθ(r, p)
    μ = ((1 - p) * r) / p
    σ2 = ((1-p) * r) / p^2
    α = μ^2 / σ2
    θ = σ2 / μ
    α, θ
end



# %% --------
function get_fits(;kws...)
    G = grid(;kws...)
    map(G) do g
        try
            x = rand(NegativeBinomial(g.thresh, g.p), 10000) .+ g.thresh
            fit(Gamma, x)
        catch
            missing
        end
    end
end

fits = get_fits(
    p = .1:.05:.9,
    thresh = 10:20:100,
)

figure() do
    plot(getfield.(fits, :α), title="α")
end
# %% --------

fits = get_fits(
    thresh = 10:2:100,
    p = .1:.1:.9,
)

figure() do
    plot(getfield.(fits, :α), title="α")
end

# α is linear with threshold, slope might be related to 1/(1-p)

# %% --------
fits = get_fits(
    p = .1:.05:.9,
    thresh = 10:5:100
)

figure() do
    p1 = heatmap(getfield.(fits, :α), title="α")
    p2 = heatmap(getfield.(fits, :θ), title="θ")
    plot(p1, p2, size=(700,300))
end
# θ depends only on p
# α depends on both p and thresh

# %% --------
θ = getfield.(fits, :θ)

figure() do
    mean(θ; dims=2) |> dropdims(2) |> plot
    x = .1:.01:.9
    plot!(x, 1 ./ x)
end

# %% --------

G = grid(
    p = .1:.05:0.9,
    n = [4, 8],
    thresh = [100],
)

fits = @showprogress map(G) do x
    x = [run_sim(x.n, x.p, x.thresh) for i in 1:10000]
    fit(Gamma, x)
end

figure() do
    plot(getfield.(fits(thresh=100), :θ))
end




# %% ==================== Expectation ====================

G = grid(
    p = .1:.02:1,
    n = [4, 8, 16],
    thresh = [100],
)

res = @showprogress map(G) do x
    run_many(x.n, x.p, x.thresh)
end

# %% --------
using LsqFit

function curve_fit_pred(f, p0, x::KeyedArray)
    k = only(axiskeys(x))
    cfit = curve_fit(f, k, x, p0)
    x .* 0 .+ f(k, cfit.param)  # hack to take the keys from x
end

# %% --------
figure() do
    thresh = 100
    for n in axiskeys(res, :n)
        xx = res(;n, thresh)
        p = axiskeys(xx, 1)
        plot!(log.(xx), ylabel="log(RT)")
        plot!(p, o@.( log(thresh / (n * p))), ls=:dot, color=:black)
    end
end

# %% --------
figure() do
    thresh = 100
    for n in axiskeys(res, :n)
        xx = res(;n, thresh)
        p = axiskeys(xx, 1)
        plot!(xx, ylabel="RT")
        plot!(p, @.(thresh / (n * p)), ls=:dot, color=:black)        
        # plot!(ft, ls=:dot, color=:black)
    end
end

# %% --------
figure() do
    for i in 1:size(res, 2)
        plot!(res[:, i, 1], ylabel="RT")
        ft = curve_fit_pred([1., 1.], res[:, i, 1]) do x, (p2,)
           @. p2 ./ x 
        end
        plot!(ft, ls=:dot, color=:black)
    end
end


# %% --------
figure() do
    for i in 1:size(res, 2)
        plot!(res[:, i, 1], ylabel="RT")
        ft = curve_fit_pred([1., 1.], res[:, i, 1]) do x, (p1,p2)
           @. p1 + p2 ./ x 
        end
        plot!(ft, ls=:dot, color=:black)
    end
end

# %% --------
G = grid(
    n = 1:10,
    thresh = [50, 100],
)

res = @showprogress map(G) do (n, thresh)
    ps = .1:.02:1
    y = run_many.(n, ps, thresh; N=100000)
    cfit = curve_fit(ps, y, [1.,]) do x, (p2,)
        @. p2 ./ x 
    end
    cfit.param
end

# %% --------
p = thresh / (n * rt)
plot(p, o@.( log(thresh / (n * p))), ls=:dot, color=:black)


