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
Problem data 1: Smooth coefficient
=#
# Problem data
# D₁(x) = @. 0.5
# f(x) = @. 1.0
# u(x) = @. x*(1-x)
# ∇u(x) = ForwardDiff.derivative(u,x)
# domain = (0.0,1.0)

#=
Problem data 2: Oscillatory diffusion coefficient
=#
# domain = (0.0,1.0)
# D₂(x) = (2 + cos(2π*x/(2^-6)))^-1
# f(x) = 0.5*π^2*sin(π*x)

#=
Problem data 3: Random diffusion coefficient
=#
domain = (0.0,1.0)
Nₑ = 2^7
nds_micro = LinRange(domain[1], domain[2], Nₑ+1)
diffusion_micro = 0.5 .+ 4.5*rand(Nₑ+1)
function _D(x::Float64, nds_micro::AbstractVector{Float64}, diffusion_micro::Vector{Float64})
  nₑ = size(nds_micro,1)
  for i=1:nₑ
    if(nds_micro[i] ≤ x ≤ nds_micro[i+1])      
      return diffusion_micro[i+1]
    else
      continue
    end 
  end
end
function D₃(x::Float64; nds_micro = nds_micro, diffusion_micro = diffusion_micro)
  _D(x, nds_micro, diffusion_micro)
end
f(x) = sin(5π*x)

#=
FEM parameters
=#
nc = 2^3 # Number of elements in the coarse space
nf = 2^15 # Number of elements in the fine space
p = 1 # Degree of polynomials in the coarse space
q = 1 # Degree of polynomials in the fine space
l = 4
quad = gausslegendre(2)
    
#=
Solve the saddle point problems to obtain the new basis functions
=#

preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p));
cache = basis_cache(q), zeros(Float64,p+1), quad, preallocated_data
compute_ms_basis!(cache, nc, q, p, D₃) 
#=
compute_ms_basis!(cache, nc, q, p, D₁) # Smooth Coefficient
compute_ms_basis!(cache, nc, q, p, D₂) # Oscillatory Coefficient
compute_ms_basis!(cache, nc, q, p, D₃) # Random Coefficient
=#
fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
nds_fineₛ, elem_fineₛ = fine
nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, global_to_patch_indices, L, Lᵀ, ipcache = patch
sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
sKms, sFms = multiscale

# Compute the full stiffness matrix on the fine scale
sKe_ϵ = zeros(Float64, q+1, q+1, nf)
sFe_ϵ = zeros(Float64, q+1, nf)
fillsKe!(sKe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, D₃) 
#=
fillsKe!(sKe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, D₁) # Smooth Coefficient
fillsKe!(sKe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, D₂) # Oscillatory Coefficient
fillsKe!(sKe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, D₃) # Random Coefficient
=#
fillLoadVec!(sFe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, f)
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]
solϵ = vcat(0,solϵ,0)
matrix_cache = split_stiffness_matrix(Kϵ, global_to_patch_indices)
vector_cache = split_load_vector(Fϵ, global_to_patch_indices)
cache = local_basis_vecs, global_to_patch_indices, L, Lᵀ, matrix_cache, ipcache
fillsKms!(sKms, cache, nc, p, l)
cache = local_basis_vecs, global_to_patch_indices, Lᵀ, vector_cache
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
plot!(plt, nds_fine, solϵ, label="Exact solution")
plt3 = plot()
plot!(plt3, nds_fine, D₃.(nds_fine), lw=2, label="Diffusion Coefficient", lc=:black)
plot!(plt3,label="Diffusion coefficient")
#=
plot!(plt3, nds_fine, D₁.(nds_fine), lc=:black, label="Diffusion coefficient")
plot!(plt3, nds_fine, D₂.(nds_fine), lc=:black, label="Diffusion coefficient")
plot!(plt3, nds_fine, D₃.(nds_fine), lc=:black, label="Diffusion coefficient")
=#
ylims!(plt3,(0,10))
plt2 = plot(plt,plt3,layout=(2,1))

## Compute the errors
l2error = 0.0
energy_error = 0.0
bc = basis_cache(q)
qs,ws=quad
for j=1:nf, jj=1:lastindex(qs)
  x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,2]])*0.5 + (0.5*nf^-1)*qs[jj]
  ϕᵢ!(bc,qs[jj])
  global l2error += ws[jj]*(dot(solϵ[elem_fine[j,:]],bc[3]) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
  ∇ϕᵢ!(bc,qs[jj])
  # Smooth Coefficient
  # global energy_error += ws[jj]*D₁(x̂)*(dot(solϵ[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
  # Oscillatory coefficient
  # global energy_error += ws[jj]*D₂(x̂)*(dot(solϵ[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
  # Random coefficient
  global energy_error += ws[jj]*D₃(x̂)*(dot(solϵ[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
end

@show l2error, energy_error