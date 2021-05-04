include("figure.jl")
include("utils.jl")
using ProgressMeter
using Distributions

#= 
Question: how many samples from a binomial does it take to reach a threshold
Conclusion: the expected RT is threshold / (p * n) ... duh!
=#


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


