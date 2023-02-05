###### ######## ######## ######## ####
# Main file containing the functions #
###### ######## ######## ######## ####
using Plots
using BenchmarkTools
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

include("basis_functions.jl")
include("assemble_matrices.jl")
include("preallocate_matrices.jl")

#=
Problem parameters
=#
# Problem data
D(x) = @. 0.5
f(x) = @. 1.0
u(x) = @. x*(1-x)
∇u(x) = ForwardDiff.derivative(u,x)
domain = (0.0,1.0)

#=
FEM parameters
=#
nc = 2^8 # Number of elements in the coarse space
nf = 2^15 # Number of elements in the fine space
p = 1 # Degree of polynomials in the coarse space
q = 1 # Degree of polynomials in the fine space
l = 10
quad = gausslegendre(2)
    
#=
Solve the saddle point problems to obtain the new basis functions
=#

preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p));
cache = basis_cache(q), zeros(Float64,p+1), quad, preallocated_data
compute_ms_basis!(cache, nc, q, p)

fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
nds_fineₛ, elem_fineₛ = fine
nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, global_to_patch_indices, ipcache = patch
sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
sKms, sFms = multiscale

# Compute the full stiffness matrix on the fine scale
sKe_ϵ = zeros(Float64, q+1, q+1, nf)
sFe_ϵ = zeros(Float64, q+1, nf)
fillsKe!(sKe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad)
fillLoadVec!(sFe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, f)
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]
solϵ = vcat(0,solϵ,0)
cache = local_basis_vecs, Kϵ, global_to_patch_indices, ipcache
fillsKms!(sKms, cache, nc, p, l)
cache = local_basis_vecs, Fϵ, global_to_patch_indices, ipcache
fillsFms!(sFms, cache, nc, p, l)

Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
Fₘₛ = zeros(Float64,nc*(p+1))
for t=1:nc
  Kₘₛ[ms_elem[t], ms_elem[t]] += sKms[t]
  Fₘₛ[ms_elem[t]] += sFms[t]
end
sol = Kₘₛ\Fₘₛ
uhsol = zeros(Float64,nf+1)
for j=1:nc, i=0:p
  uhsol[:] += sol[(p+1)*j+i-p]*local_basis_vecs[j][:,i+1]
end

plt = plot(nds_fine, uhsol, label="Approximate solution")
plot!(plt, nds_fine, u.(nds_fine), label="Exact solution")

## Compute the errors
usol = u.(nds_fine)
l2error = 0.0
energy_error = 0.0
bc = basis_cache(q)
qs,ws=quad
for j=1:nf, jj=1:lastindex(qs)
  x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,1]])*0.5 + (0.5*nf^-1)*qs[jj]
  ϕᵢ!(bc,qs[jj])
  global l2error += ws[jj]*(dot(solϵ[elem_fine[j,:]],bc[3]) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
  ∇ϕᵢ!(bc,qs[jj])
  global energy_error += ws[jj]*D(x̂)*(dot(solϵ[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
end

@show l2error, energy_error