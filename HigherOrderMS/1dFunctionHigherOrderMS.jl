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
nodes, elems = mesh((0,1), 2^5)
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
# Get the element connectivity
ms_elem_matrix = new_connectivity_matrices(elems, p, l) 
tn = p+1:maximum(ms_elem_matrix)-(p)
bn = vcat(1:p+1, maximum(ms_elem_matrix)-(p):maximum(ms_elem_matrix))
fn = setdiff(tn, bn)
# Compute the local basis
RˡₕΛₖ = compute_ms_basis(nodes, elems, A, fespace, l; Nfine=2^9, qorder=10)
# Assemble the multiscale system
K,M,F = assemble_matrix_MS_MS(nodes, elems, ms_elem_matrix, RˡₕΛₖ, A, x-> π^2*sin(π*x), fespace; qorder=qorder, Nfine=Nfine, plot_basis=5)
# Solve the linear system (zero BC)
sol = K[fn,fn]\F[fn]
sol = vcat(repeat([0],p+1), sol, repeat([0],p+1))
# Plot the solution
xvals = 0:0.01:1
uhxvals = map(x -> uᵐˢₕ(x, sol, RˡₕΛₖ, nodes, elems, fespace),xvals)
plt = plot(xvals,uhxvals)