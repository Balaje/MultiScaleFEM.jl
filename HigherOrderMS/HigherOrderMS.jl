using FastGaussQuadrature
using LinearAlgebra
using Plots
using SparseArrays   
using LoopVectorization
using SparseArrays
using Gridap
using FillArrays
using BenchmarkTools
using Gridap.Arrays
using LazyArrays

include("coarse_to_fine.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("multiscale_basis-functions.jl")