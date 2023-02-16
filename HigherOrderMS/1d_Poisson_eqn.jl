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
nc = 2^2 # Number of elements in the coarse space
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
nds_coarse, elems_coarse, nds_fine, elem_fine = fullspace[1:4]
patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch[3:7]
ms_elem = assems[3]
sKms, sFms = multiscale
bc = basis_cache(q)

# Compute the full stiffness matrix on the fine scale
assem_cache = assembler_cache(nds_fine, elem_fine, quad, q)
fillsKe!(assem_cache, D₃)
fillsFe!(assem_cache, f)
#=
fillsKe!(assem_cache, D₁) # Smooth Coefficient
fillsKe!(assem_cache D₂) # Oscillatory Coefficient
fillsKe!(assem_cache, D₃) # Random Coefficient
=#
Kϵ = sparse(assem_cache[5][1], assem_cache[5][2], assem_cache[5][3])
Fϵ = collect(sparsevec(assem_cache[6][1],assem_cache[6][2]))
solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]
solϵ = vcat(0,solϵ,0)
contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
matrix_cache = mat_contribs!(contrib_cache, D₃)
#=
matrix_cache = mat_contribs!(contrib_cache, D₁)
matrix_cache = mat_contribs!(contrib_cache, D₂)
matrix_cache = mat_contribs!(contrib_cache, D₃)
=#
vector_cache = vec_contribs!(contrib_cache, f)
cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
fillsKms!(sKms, cache, nc, p, l)
cache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
fillsFms!(sFms, cache, nc, p, l)

Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
Fₘₛ = zeros(Float64,nc*(p+1))
cache = Kₘₛ, Fₘₛ
assemble_MS!(cache, sKms, sFms, ms_elem)
sol = Kₘₛ\Fₘₛ
uhsol = zeros(Float64,nf+1)
sol_cache = similar(uhsol)
cache = uhsol, sol_cache
build_solution!(cache, sol, local_basis_vecs)

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