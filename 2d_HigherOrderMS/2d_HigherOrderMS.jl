using Gridap
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData

using StaticArrays
using NearestNeighbors
using SparseArrays

using PyPlot
using Plots
pyplot()
using LaTeXStrings
using ColorSchemes
PyPlot.matplotlib[:rc]("text", usetex=true) 
PyPlot.matplotlib[:rc]("mathtext",fontset="cm")
PyPlot.matplotlib[:rc]("font",family="serif",size=20)

using SplitApplyCombine
using LinearAlgebra

include("assemble_matrices.jl")
include("coarse_to_fine_map.jl")
include("new_coarse_to_fine.jl")
include("multiscale_basis-functions.jl")
