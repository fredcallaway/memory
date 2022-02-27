using Serialization
using AxisKeys
using SplitApplyCombine
import Base.Iterators: product
using Statistics

# %% ==================== Project-specific ====================

# length of each presentation
function parse_presentations(cs, ms_per_sample)
    f = 0
    x = Int[]
    for (i, c) in enumerate(cs)
        if c != f
            push!(x, 0)
            f = c
        end
        x[end] += ms_per_sample
    end
    x
end


# %% ==================== General Purpose ====================


function cache(f, file; disable=false)
    disable && return f()
    isfile(file) && return deserialize(file)
    result = f()
    mkpath(dirname(file))
    serialize(file, result)
    result
end

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

function grid(;kws...)
    X = map(Iterators.product(values(kws)...)) do x
        (; zip(keys(kws), x)...)
    end
    KeyedArray(X; kws...)
end

function keyed(name, xs)
    KeyedArray(xs; Dict(name => xs)...)
end

keymax(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmax(X).I))...)
keymax(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmax(x)]
keymin(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmin(X).I))...)
keymin(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmin(x)]

Base.dropdims(idx::Union{Symbol,Int}...) = X -> dropdims(X, dims=idx)
squeezify(f) = (X, dims...) -> dropdims(f(X; dims); dims)
smaximum = squeezify(maximum)
sminimum = squeezify(minimum)
smean = squeezify(mean)
ssum = squeezify(sum)
ssum = squeezify(sum)

function monte_carlo(f, N=10000)
    N \ mapreduce(+, 1:N) do i
        f()
    end
end

function repeatedly(f, N=10000)
    map(1:N) do i
        f()
    end
end

linscale(x, low, high) = low + x * (high-low)
logscale(x, low, high) = exp(log(low) + x * (log(high) - log(low)))
unlinscale(x, low, high) = (x - low) / (high-low)
unlogscale(x, low, high) = (log(x) - log(low)) / (log(high) - log(low))

juxt(fs...) = x -> Tuple(f(x) for f in fs)

nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)
normalize(x) = x ./ sum(x)