using Serialization
using AxisKeys
using SplitApplyCombine
import Base.Iterators: product
using Statistics
using DataFrames, DataFramesMeta, CSV
using Printf

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

fillnan(x, repl=0.) = isnan(x) ? repl : x
robust_median(x; n=30) = length(x) < n ? NaN : median(x)


# %% ==================== General Purpose ====================

function print_header(txt; color=:magenta)
    display_width = displaysize(stdout)[2]
    n_fill = fld(display_width - length(txt) - 2, 2)
    n_space = 2
    n_dash = n_fill - n_space
    print(' '^n_space)
    printstyled('-'^n_dash; color, bold=true)
    print(' ', txt, ' ')
    printstyled('-'^n_dash; color, bold=true)
    print(' '^n_space)
    print("\n")
end

macro catch_missing(expr)
    esc(quote
        try
            $expr
        catch
            missing
        end
    end)
end

function pooled_mean_std(ns::AbstractVector{<:Integer},
                        μs::AbstractVector{<:Number},
                        σs::AbstractVector{<:Number})
    nsum = sum(ns)
    meanc = ns' * μs / nsum
    vs = replace!(σs .^ 2, NaN=>0)
    varc = sum((ns .- 1) .* vs + ns .* abs2.(μs .- meanc)) / (nsum - 1)
    return meanc, .√(varc)
end

function cache(f, file; disable=false, read_only=false, overwrite=false)
    disable && return f()
    !overwrite && isfile(file) && return deserialize(file)
    read_only && error("No cached result $file")
    result = f()
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

function initialize_keyed(val; keys...)
    KeyedArray(fill(val, (length(v) for (k, v) in keys)...); keys...)
end

function wrap_counts(df::DataFrame; dims...)
    @chain df begin
        groupby(collect(keys(dims)))
        combine(nrow => :n)
        AxisKeys.populate!(initialize_keyed(0.; dims...), _, :n)
    end
end

function wrap_pivot(df::DataFrame, val, f; dims...) 
    @chain df begin
        groupby(collect(keys(dims)))
        combine(val => f => :_val)
        AxisKeys.populate!(initialize_keyed(0.; dims...), _, :_val)
    end
end

macro bywrap(x, what, val, default=missing)
    arg = :(:_val = $val)
    esc(quote
        b = $(DataFramesMeta.by_helper(x, what, arg))
        what_ = $what isa Symbol ? ($what,) : $what
        wrapdims(b, :_val, what_..., sort=true; default=$default)
    end)
end

function keyed(name, xs)
    KeyedArray(xs; Dict(name => xs)...)
end

keymax(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmax(X).I))...)
keymax(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmax(x)]
keymin(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmin(X).I))...)
keymin(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmin(x)]

round1(x) = round(x; digits=1)
round2(x) = round(x; digits=2)
round3(x) = round(x; digits=3)
round4(x) = round(x; digits=4)

fmt(digits, x) = Printf.format(Printf.Format("%.$(digits)f"), x)

function Base.diff(K::KeyedArray; dims, removefirst::Bool=true)
    range = removefirst ? (2:size(K, dims)) : (1:size(K,dims)-1)
    out = similar(selectdim(K, dims, range) )
    out[:] = Base.diff(parent(parent(K)); dims=AxisKeys.dim(parent(K),dims))
    return out
end

Base.dropdims(idx::Union{Symbol,Int}...) = X -> dropdims(X, dims=idx)
squeezify(f) = (X, dims...) -> dropdims(f(X; dims); dims)
smaximum = squeezify(maximum)
sminimum = squeezify(minimum)
smean = squeezify(mean)
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
clip(x, lo, hi) = min(hi, max(lo, x))

nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)
normalize(x) = x ./ sum(x)
normalize!(x) = x ./= sum(x)