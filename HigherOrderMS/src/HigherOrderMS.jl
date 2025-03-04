using Pkg
Pkg.add("FastGaussQuadrature")
Pkg.add("DoubleFloats")
Pkg.add("LaTeXStrings")
Pkg.add("ColorSchemes")
Pkg.add("Gridap")
Pkg.add("DoubleFloats")
Pkg.add("Plots")

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays   
using Gridap
using DoubleFloats

using LaTeXStrings
using ColorSchemes


include("coarse_to_fine.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("multiscale_basis-functions.jl")
include("time-dependent.jl")
include("corrected_basis.jl");
