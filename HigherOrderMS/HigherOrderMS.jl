using FastGaussQuadrature
using LinearAlgebra
using Plots
using SparseArrays   
using LoopVectorization
using LazyArrays
using SparseArrays
using Gridap
using FillArrays
using BenchmarkTools

include("preallocation.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("multiscale_basis-functions.jl")