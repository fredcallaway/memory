using Pkg

packages = """
AxisKeys
CSV
DataFrames
DataFramesMeta
DataStructures
Dates
Distributions
GLM
JuliennedArrays
Optim
Parameters
Printf
ProgressMeter
Random
Serialization
Sobol
SplitApplyCombine
Statistics
StatsBase
StatsFuns
StatsPlots
"""
Pkg.add(split(packages))
Pkg.precompile()
