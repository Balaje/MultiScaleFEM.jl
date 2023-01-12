##################################################################
# Modules used to implement the Higher Order Multiscale Methods
##################################################################
using ForwardDiff
using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using Plots

include("mesh_conn.jl")
include("solve_local_problems.jl")
include("basis_functions.jl")
include("local_matrix_vector.jl")
include("assemble_matrices.jl")
include("basis_functions.jl")
include("multiscale_function.jl")

# Solve the multiscale FEM problem
#ε = 1
#A(x) = @. (2 + cos(2π*x/ε))^-1

A(x) = @. 1
# Define the coarse mesh
nodes, elems = mesh((0,1), 1)
H = nodes[2]-nodes[1]
# Number of points in the fine mesh
Nfine = 2^6
qorder = 3
# Define some parameters
p = 1 # Order of the method i.e., degree of the L² subspace
q = 1 # Polynomial degree of the fine mesh
l = 1 # Size of the element patch
fespace = (q,p) # FESpace pair (H¹, L²)
nel = size(elems,1)
Rˡₕ(x->1, A, (nodes[1],nodes[2]), nodes, elems;
    fespace=fespace, N=Nfine, qorder=qorder)
