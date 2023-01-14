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
ε = 1e-6
A(x) = @. (2 + cos(2π*x/ε))^-1

#A(x) = @. 1
f(x) = @. 1
# Define the coarse mesh
nodes, elems = mesh((0,1), 10)
H = nodes[2]-nodes[1]
# Number of points in the fine mesh
Nfine = 10^3
qorder = 10
# Define some parameters
p = 1 # Order of the method i.e., degree of the L² subspace
q = 1 # Polynomial degree of the fine mesh
l = 5 # Size of the element patch
fespace = (q,p) # FESpace pair (H¹, L²)
nel = size(elems,1)
RˡₕΛₖ = compute_ms_basis(nodes, elems, A, fespace, l; Nfine = Nfine, qorder=qorder)

# Now plot and check the basis function at i,j ∈ [1:el],[1:p+1]
plt = plot()
plt1 = plot()
for el=[5]
  for k=1:p+1
    Rₖ = RˡₕΛₖ[el,k]
    plot!(plt, Rₖ.nds, Rₖ.Λ⃗, label="Basis Function "*string(k), lw=0.5, legend=:outertopright)
    # plot!(plt, LinRange(Rₖ.nds[1], Rₖ.nds[end], 40),
    #       map(x->Λ̃ₖˡ(x, Rₖ, fespace), LinRange(Rₖ.nds[1], Rₖ.nds[end], 40)),
    #       label="", ls=:dot, lw=3, lc=:black)
  end
end
xlims!(plt,(0,1))

# ############################################
# # Solve the MS problem
# ############################################
ms_elems = new_connectivity_matrices(elems, p, l)
# K,M,F = assemble_matrix_MS_MS(nodes, elems, ms_elems, RˡₕΛₖ,
#                               A, f, fespace; qorder=qorder)
