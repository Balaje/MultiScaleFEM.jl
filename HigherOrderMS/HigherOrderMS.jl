using FastGaussQuadrature
using LinearAlgebra
using Plots
using SparseArrays   
using Gridap
using BenchmarkTools


include("coarse_to_fine.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("multiscale_basis-functions.jl")