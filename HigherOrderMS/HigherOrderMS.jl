using FastGaussQuadrature
using LinearAlgebra
using SparseArrays   
using Gridap
using BenchmarkTools

using PyPlot
using Plots
pyplot()
using LaTeXStrings
using ColorSchemes
PyPlot.matplotlib[:rc]("text", usetex=true) 
PyPlot.matplotlib[:rc]("mathtext",fontset="cm")
PyPlot.matplotlib[:rc]("font",family="serif",size=20)

include("coarse_to_fine.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("multiscale_basis-functions.jl")
include("time-dependent.jl")
include("./schur_complement.jl")