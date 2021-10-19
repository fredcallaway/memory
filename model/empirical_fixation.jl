using JSON
using DataFrames
using Memoize

# %% --------

mutable struct RandomCommitmentPolicy{D<:Distribution,E<:Distribution} <: Policy 
    m::MetaMDP
    duration_dist::D
    commitment_dist::E
    time_to_switch::Int
    time_to_commit::Int
end

function act(pol::RandomCommitmentPolicy, b::Belief)
    if b.n_step == 0
        pol.time_to_commit = rand(pol.commitment_dist)
        @assert pol.time_to_commit > 0
    end
    if b.n_step == 0 || pol.time_to_switch == 0
        pol.time_to_commit -= 1
        commit = pol.time_to_commit == 0
        pol.time_to_switch = commit ? -1 : ceil(Int, rand(pol.duration_dist)) - 1
        #@assert pol.time_to_switch >= 0
        rand(setdiff(1:n_item(b), b.focused))
    else
        pol.time_to_switch -= 1
        b.focused
    end
end

@memoize function load_multi()
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

function random_policy(m; fixations=load_multi().presentation_times, commitment=true, skip_final=commitment)
    ms_per_sample = 1000 * (TIME_LIMIT / m.max_step)

    commitment_dist = if commitment
        n_fix = filter!(x->x>0, length.(fixations))
        fit(DiscreteNonParametric, n_fix)
    else
        DiscreteNonParametric([100000], [1.])
    end

    durations = mapreduce(vcat, fixations) do trial_fix
        if skip_final
            trial_fix = trial_fix[1:end-1]
        end
        length(trial_fix) == 0 && return Int[]
        max.(1, round.(Int, trial_fix ./ ms_per_sample))
    end
    duration_dist = fit(DiscreteNonParametric, durations)
    RandomCommitmentPolicy(m, duration_dist, commitment_dist, 0, 0)
end

function individual_empirical_policies(m; kws...)
    df = load_multi()
    pols = map(groupby(df, :wid)) do d
        random_policy(m; fixations=d.presentation_times, kws...)
    end
    filter!(pols) do pol
        length(pol.duration_dist.support) != 0
    end
end
