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
f(x) = @. (π)^2*sin(π*x)
u(x) = @. sin(π*x)
∇u(x) = @. π*cos(π*x)
domain = (0.0,1.0)

"""
Function to compute the l2 and energy errors
"""

#=
Constant paramters
=#
p = 2
q = 1
nf = 2^16 # Size of the background mesh
qorder = 3
quad = gausslegendre(qorder)

𝒩 = [1,2,4,8,16,32,64,128,256]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

plt = plot()
plt1 = plot()
plt2 = plot()

# Build the matrices for the fine scale problem
nds_fine = LinRange(domain[1], domain[2], nf+1)
elem_fine = [elem_conn(i,j) for i=1:nf, j=0:1]
assem_H¹H¹ = ([H¹Conn(q,i,j) for _=0:q, j=0:q, i=1:nf],
[H¹Conn(q,i,j) for j=0:q, _=0:q, i=1:nf], 
[H¹Conn(q,i,j) for j=0:q, i=1:nf])
# Fill the final-scale matrix vector system
sKe_ϵ = zeros(Float64, q+1, q+1, nf)
sFe_ϵ = zeros(Float64, q+1, nf)
fillsKe!(sKe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad)
fillLoadVec!(sFe_ϵ, basis_cache(q), nds_fine, elem_fine, q, quad, f)
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
solϵ = Kϵ[2:nf,2:nf]\Fϵ[2:nf]
solϵ = vcat(0,solϵ,0)

for l in [4,5,6,7,8,9,10]
  fill!(L²Error,0.0)
  fill!(H¹Error,0.0)
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    local preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
    local cache = basis_cache(q), zeros(Float64,p+1), quad, preallocated_data
    compute_ms_basis!(cache, nc, q, p)
    
    local fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
    local nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
    local nds_fineₛ, elem_fineₛ = fine
    local nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, global_to_patch_indices, ipcache = patch
    local sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
    local assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
    local sKms, sFms = multiscale
    
    # Compute the full stiffness matrix on the fine scale    
    local cache = local_basis_vecs, Kϵ, global_to_patch_indices, ipcache
    fillsKms!(sKms, cache, nc, p, l)
    local cache = local_basis_vecs, Fϵ, global_to_patch_indices, ipcache
    fillsFms!(sFms, cache, nc, p, l)
    
    local Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
    local Fₘₛ = zeros(Float64,nc*(p+1))
    for t=1:nc
      Kₘₛ[ms_elem[t], ms_elem[t]] += sKms[t]
      Fₘₛ[ms_elem[t]] += sFms[t]
    end
    local sol = Kₘₛ\Fₘₛ
    local uhsol = zeros(Float64,nf+1)
    for j=1:nc, i=0:p      
      uhsol[:] += sol[(p+1)*j+i-p]*local_basis_vecs[j][:,i+1]
    end
    
    ## Compute the errors
    local usol = u.(nds_fine)
    local bc = basis_cache(q)
    local qs,ws=quad    
    for j=1:nf, jj=1:lastindex(qs)
      x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,1]])*0.5 + (0.5*nf^-1)*qs[jj]
      ϕᵢ!(bc,qs[jj])
      L²Error[itr] += ws[jj]*(dot(solϵ[elem_fine[j,:]],bc[3]) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
      ∇ϕᵢ!(bc,qs[jj])
      H¹Error[itr] += ws[jj]*D(x̂)*(dot(solϵ[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
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

plot!(plt2, 0:0.01:1, u.(0:0.01:1), label="Exact", lw=1, lc=:black)

#=
Sample Error and Rate: (p=2, l=10)
julia> L²Error
9-element Vector{Float64}:
 0.0005725950183078252
 0.00014407512704031465
 3.6711256956560554e-6
 1.0788626238677463e-7
 3.3162425566359246e-9
 2.1691680711022863e-10
 2.3197636023288994e-10
 2.401925445565128e-10
 2.3613047517807476e-10

julia> H¹Error
9-element Vector{Float64}:
 0.008713475989892316
 0.0031616953858213323
 0.00017571582397748589
 1.060883432940712e-5
 6.56667777486306e-7
 1.0111154545588088e-7
 2.449145579205192e-7
 5.308265132299848e-7
 1.086597557970863e-6

julia> log.(L²Error[2:end]./L²Error[1:end-1])./log.(𝒩[1:end-1]./𝒩[2:end])
8-element Vector{Float64}:
  1.9906938268117103
  5.294454969904888
  5.0886394345283925
  5.0238178295422795
  3.9343357872042715
 -0.0968359536745811
 -0.050213577622697125
  0.024607122385724253

julia> log.(H¹Error[2:end]./H¹Error[1:end-1])./log.(𝒩[1:end-1]./𝒩[2:end])
8-element Vector{Float64}:
  1.462549977570873
  4.1693823543479756
  4.049906067619909
  4.01395867091521
  2.699215923047742
 -1.276330790151233
 -1.1159618994832707
 -1.0335053737865008
=#