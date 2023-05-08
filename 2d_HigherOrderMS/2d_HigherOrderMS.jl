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
using Plots

include("assemble_matrices.jl")
include("coarse_to_fine_map.jl")
include("multiscale_basis-functions.jl")
