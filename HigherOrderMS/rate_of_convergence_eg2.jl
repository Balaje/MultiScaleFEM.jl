######### ############ ############ ############ ###########
# Compute the rate of convergence of the multiscale method
# With Oscillatory Diffusion Coefficient 
# (-) D(x) = (2 + cos(2π*x/(2^-6)))^-1
#     f(x) = 0.5*π^2*sin(π*x)  
######### ############ ############ ############ ###########
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
Problem data 2: Oscillatory diffusion coefficient
=#
domain = (0.0,1.0)
D₂(x) = (2 + cos(2π*x/(2^-6)))^-1
f(x) = 0.5*π^2*sin(π*x)

#=
Constant paramters
=#
p = 1
q = 1
nf = 2^16 # Size of the background mesh
qorder = 3
quad = gausslegendre(qorder)

𝒩 = [1,2,4,8,16,32,64,128]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

plt = plot()
plt1 = plot()

# Build the matrices for the fine scale problem
nds_fine = LinRange(domain[1], domain[2], nf+1)
elem_fine = [elem_conn(i,j) for i=1:nf, j=0:1]
assem_H¹H¹ = ([H¹Conn(q,i,j) for _=0:q, j=0:q, i=1:nf],
[H¹Conn(q,i,j) for j=0:q, _=0:q, i=1:nf], 
[H¹Conn(q,i,j) for j=0:q, i=1:nf])
# Fill the fine-scale matrix vector system
assem_cache = assembler_cache(nds_fine, elem_fine, quad, q)
fillsKe!(assem_cache, D₂)
fillsFe!(assem_cache, f)
Kϵ = sparse(assem_cache[5][1], assem_cache[5][2], assem_cache[5][3])
Fϵ = collect(sparsevec(assem_cache[6][1],assem_cache[6][2]))
solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]
solϵ = vcat(0,solϵ,0)

# for l in [4,5,6,7,8,9,10]
for l in [4,5,6,7,8,9]
  fill!(L²Error,0.0)
  fill!(H¹Error,0.0)
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    local preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
    local cache = basis_cache(q), zeros(Float64,p+1), quad, preallocated_data
    compute_ms_basis!(cache, nc, q, p, D₂)
    
    local fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
    local nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
    local nds_fineₛ, elem_fineₛ = fine
    local nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch
    local sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
    local assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
    local sKms, sFms = multiscale
    
    # Compute the full stiffness matrix on the fine scale    
    local contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
    local matrix_cache = mat_contribs!(contrib_cache, D₂)
    local vector_cache = vec_contribs!(contrib_cache, f)
    local cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
    fillsKms!(sKms, cache, nc, p, l)
    local cache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
    fillsFms!(sFms, cache, nc, p, l)
    
    local Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
    local Fₘₛ = zeros(Float64,nc*(p+1))
    local cache = Kₘₛ, Fₘₛ
    assemble_MS!(cache, sKms, sFms, ms_elem)
    local sol = Kₘₛ\Fₘₛ
    local uhsol = zeros(Float64,nf+1)
    local sol_cache = similar(uhsol)
    local cache = uhsol, sol_cache
    build_solution!(cache, sol, local_basis_vecs)
    uhsol, _ = cache
    
    ## Compute the errors
    # local usol = u.(nds_fine)
    local bc = basis_cache(q)
    local qs,ws=quad    
    for j=1:nf, jj=1:lastindex(qs)
      x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,2]])*0.5 + (0.5*nf^-1)*qs[jj]
      ϕᵢ!(bc,qs[jj])
      L²Error[itr] += ws[jj]*(dot(solϵ[elem_fine[j,:]],bc[3]) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
      ∇ϕᵢ!(bc,qs[jj])
      H¹Error[itr] += ws[jj]*D₂(x̂)*(dot(solϵ[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
    end    

    L²Error[itr] = sqrt(L²Error[itr])
    H¹Error[itr] = sqrt(H¹Error[itr])
        
    println("Done nc = "*string(nc))
  end
  
  println("Done l = "*string(l))
  plot!(plt, 1 ./𝒩, L²Error, label="(p="*string(p)*"), L² (l="*string(l)*")", lw=2)
  plot!(plt1, 1 ./𝒩, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=2)
  scatter!(plt, 1 ./𝒩, L²Error, label="", markersize=2)
  scatter!(plt1, 1 ./𝒩, H¹Error, label="", markersize=2, legend=:best)
end

plot!(plt1, 1 ./𝒩, (1 ./𝒩).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)
plot!(plt, 1 ./𝒩, (1 ./𝒩).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)