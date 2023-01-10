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
nodes, elems = mesh((0,1), 4)
H = nodes[2]-nodes[1]
# Number of points in the fine mesh
Nfine = 2^9
qorder = 10
# Define some parameters
p = 1 # Order of the method i.e., degree of the L² subspace
q = 4 # Polynomial degree of the fine mesh
l = 1 # Size of the element patch
fespace = (q,p) # FESpace pair (H¹, L²)
nel = size(elems,1)
# Get the element connectivity
ms_elem_matrix = new_connectivity_matrices(elems, p, l) 
tn = 1:maximum(ms_elem_matrix)
bn = vcat(1, maximum(ms_elem_matrix)-p)
fn = setdiff(tn, bn)
# Compute the local basis
RˡₕΛₖ = compute_ms_basis(nodes, elems, A, fespace, l; Nfine=Nfine, qorder=qorder)
# Plot the basis
plt = plot()
for el=1:nel  
  for  i=1:p+1
    Rₖ = RˡₕΛₖ[el,i]
    xs = getindex.(Rₖ.Ω.grid.node_coords,1)        
    fxs = map(x->Λ̃ₖˡ(x, Rₖ), xs) 
    plot!(plt, xs, fxs, lw=2, label="Basis "*string(i)*" in element "*string(el), legend=:outertopleft)
  end
end
display(plt)


"""
In progress:
"""
# # Assemble the multiscale system
# K,M,F = assemble_matrix_MS_MS(nodes, elems, ms_elem_matrix, RˡₕΛₖ, A, x-> sin(π*x), fespace; qorder=qorder, Nfine=Nfine, plot_basis=4)
# # # Solve the linear system (zero BC)
# sol = K[fn,fn]\F[fn]
# sol = vcat(0, sol[1:end-p], 0, sol[end-p+1:end])
# # # Plot the solution
# xvals = 0:0.01:1
# fxvals = map(x->uᵐˢₕ(x, sol, RˡₕΛₖ, elems), xvals)
# # plt = plot(xvals,uhxvals)