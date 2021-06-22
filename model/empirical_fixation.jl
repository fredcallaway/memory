using JSON
using DataFrames
using Memoize

@memoize function load_fixations()
    versions = ["v3.4", "v3.5", "v3.6"]
    multi = mapreduce(vcat, versions) do version
        CSV.read("../data/$version/multi-recall.csv", DataFrame)
    end
    filter!(multi) do row
        !row.practice
    end
    # TODO: throw out bad trials
    Vector{Vector{Float64}}(map(JSON.parse, multi.presentation_times))
end

function empirical_policy(m)
    time_limit = 15
    fixations = load_fixations()
    ms_per_sample = 1000 * (time_limit / m.max_step)
    x = round.(Int, reduce(vcat, fixations) ./ ms_per_sample)
    x = max.(x, 1)
    d = fit(DiscreteNonParametric, x)
    SwitchDistributionPolicy(m, d)
end

function get_last_other(df)
    last_ = Int[]; other = Int[]
    for pt in df.presentation_times
        if !isempty(pt)
            push!(last_, pt[end])
            push!(other, pt[1:end-1]...)
        end
    end
    mean(last_), mean(other)
end
