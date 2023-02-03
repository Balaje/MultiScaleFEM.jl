######### ############ ############ ############ ###########
# Compute the rate of convergence of the multiscale method
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
Problem data
=#
D(x) = @. 1.0
f(x) = @. (5π)^2*sin(5π*x)
u(x) = @. sin(5π*x)
∇u(x) = @. 5π*cos(5π*x)
domain = (0.0,1.0)

"""
Function to compute the l2 and energy errors
"""

#=
Constant paramters
=#
p = 1
q = 1
nf = 2^11 # Size of the background mesh
qorder = 2
quad = gausslegendre(qorder)

#𝒩 = [1,2,4,8,16,32,64,128]
𝒩 = [2,4,8,16,32]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

#plt = plot()
#plt1 = plot()
plt2 = plot()

for l in 𝒩
  fill!(L²Error,0.0)
  fill!(H¹Error,0.0)
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    local preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
    local cache = basis_cache(q), zeros(Float64,p+1), quad, preallocated_data
    compute_ms_basis!(cache, nc, q, p)
    
    local fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
    local nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
    local nds_fineₛ, elem_fineₛ = fine
    local nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, ipcache = patch
    local sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
    local assem_H¹H¹ₛ, assem_H¹L²ₛ = assems
    local sKms, sFms = multiscale
    
    # Compute the full stiffness matrix on the fine scale
    local sKe_ϵ = zeros(Float64, q+1, q+1, nf)
    local sFe_ϵ = zeros(Float64, q+1, nf)
    fillsKe!(sKe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad)
    fillLoadVec!(sFe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, f)
    local Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
    local Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
    local cache = local_basis_vecs, Kϵ, patch_indices_to_global_indices, ipcache
    fillsKms!(sKms, cache, nc, p, l)
    local cache = local_basis_vecs, Fϵ, patch_indices_to_global_indices, ipcache
    fillsFms!(sFms, cache, nc, p, l)
    
    local Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
    local Fₘₛ = zeros(Float64,nc*(p+1))
    local ms_elem = Vector{Vector{Int64}}(undef,nc)
    for t=1:nc
      start = max(1,t-l)*(p+1)-1
      last = min(nc,t+l)*(p+1)
      ms_elem[t] = start:last
      Kₘₛ[ms_elem[t], ms_elem[t]] += sKms[t]
      Fₘₛ[ms_elem[t]] += sFms[t]
    end
    local sol = Kₘₛ\Fₘₛ
    local uhsolₛ = similar(nds_fineₛ)
    local uhsol = zeros(Float64,nf+1)
    for j=1:nc, i=0:p
      uhsolₛ[j] = zeros(Float64,size(nds_fineₛ[j]))
      uhsol[patch_indices_to_global_indices[j]] += sol[(p+1)*j+i-p]*local_basis_vecs[j][:,i+1]
    end
    
    ## Compute the errors
    local usol = u.(nds_fine)
    local bc = basis_cache(q)
    local qs,ws=quad    
    for j=1:nf, jj=1:lastindex(qs)
      x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,1]])*0.5 + (0.5*nf^-1)*qs[jj]
      ϕᵢ!(bc,qs[jj])
      L²Error[itr] += ws[jj]*(u(x̂) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
      ∇ϕᵢ!(bc,qs[jj])
      H¹Error[itr] += ws[jj]*D(x̂)*(∇u(x̂) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
    end    
        
    println("Done nc = "*string(nc))
  end
  
  println("Done l = "*string(l))
  plot!(plt, 1 ./𝒩, L²Error, label="L² (l="*string(l)*")", lw=2)
  plot!(plt1, 1 ./𝒩, H¹Error, label="Energy (l="*string(l)*")", lw=2)
  scatter!(plt, 1 ./𝒩, L²Error, label="", markersize=2)
  scatter!(plt1, 1 ./𝒩, H¹Error, label="", markersize=2, legend=:best)
end

plot!(plt1, 1 ./𝒩, (1 ./𝒩).^3, label="Order 3", ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)
plot!(plt, 1 ./𝒩, (1 ./𝒩).^4, label="Order 4", ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)

plot!(plt2, 0:0.01:1, u.(0:0.01:1), label="Exact", lw=1, lc=:black)