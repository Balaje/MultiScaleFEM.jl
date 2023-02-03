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

#=
Problem data
=#
D(x) = @. 1.0
f(x) = @. π^2*sin(π*x)
u(x) = @. sin(π*x)
∇u(x) = @. π*cos(π*x)
domain = (0.0,1.0)

"""
Function to compute the l2 and energy errors
"""
function error!(cache, ue::Function, ∇ue::Function, uh::AbstractVector{Float64}, nds_fine::AbstractVector{Float64},
  elem_fine::AbstractMatrix{Int64}, quad::Tuple{Vector{Float64},Vector{Float64}})
  qs, ws = quad
  l2err, h1err, bc = cache
  l2err = 0.0; h1err = 0.0
  nf = size(elem,1)
  for t=1:nf, i=1:lastindex(qs)    
    cs = view(nds_fine,view(elem_fine,t,:))
    uhsol = view(uh,view(elem_fine,t,:))
    x̂ = (cs[2]+cs[1])*0.5 + (cs[2]-cs[1])*0.5(qs[i])
    ϕᵢ!(bc,qs[i])
    l2err += ws[i]*(ue(x̂) - dot(uhsol, bc[3]))^2*(cs[2]-cs[1])*0.5
    ∇ϕᵢ!(bc,qs[i])
    h1err += ws[i]*D(x̂)*(∇ue(x̂) - dot(uhsol, bc[3])*(2/(cs[2]-cs[1])))^2*(cs[2]-cs[1])*0.5
  end
  sqrt(l2err), sqrt(h1err)
end

#=
Constant paramters
=#
p = 1
q = 1
nf = 2^11 # Size of the background mesh
qorder = 2
quad = gausslegendre(qorder)

𝒩 = [1,2,4,8,16,32,64]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

plt = plot()
plt1 = plot()
plt2 = plot()

# The stiffness matrix and the load vector is constucted in the fine scale.
assem_H¹H¹ = ([(q)*t+ti-(q-1) for _=0:q, ti=0:q, t=1:nf], 
              [(q)*t+tj-(q-1) for tj=0:q, _=0:q, t=1:nf], 
              [(q)*t+ti-(q-1) for ti=0:q, t=1:nf])  
sKe = zeros(Float64,q+1,q+1,nf) 
elem_fine = [i+j for i=1:nf, j=0:1]
sKeϵ = zeros(Float64,q+1,q+1,nf)
sFeϵ = zeros(Float64,q+1,nf)
nds_fine = LinRange(domain[1],domain[2],nf+1)    
fillsKe!(sKeϵ, basis_cache(q), nds_fine, elem_fine, q, quad)
fillLoadVec!(sFeϵ, basis_cache(q), nds_fine, elem_fine, q, quad, f)
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKeϵ))
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFeϵ)))

for l in 𝒩
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    #=
    Precompute all the caches. Essential for computing the solution quickly
    =#
    local npatch = min(2l+1,nc)
    # Construct the coarse mesh
    local H = (domain[2]-domain[1])/nc
    local nds_coarse = domain[1]:H:domain[2]
    local tree = KDTree(nds_coarse')            
    # Connectivity of the coarse domain
    local elem_coarse = [i+j for i=1:nc, j=0:1]
    # Store the data for solving the multiscale problems
    local MS_Basis = Array{Float64}(undef,q*nf+1,nc,p+1)
    # Store the KDTrees for the patch domain
    local KDTrees = Vector{KDTree}(undef,nc)    
        
    #=
    Efficiently compute the solution to the saddle point problems.
    =#
    local cache_q = basis_cache(q)
    local cache_p = Vector{Float64}(undef,p+1)
    local cache = sKe, assem_H¹H¹, MS_Basis, cache_q, cache_p, KDTrees, q, p
    compute_ms_basis!(cache, domain, nds_coarse, elem_coarse, elem_fine, D, l)
    
    #=
      Now solve the multiscale problems.
        The multiscale basis are stored in the variable MS_Basis
    =#    
    # Some preallocated variables for the multiscale problems
    local ndofs = npatch*(p+1)
    local elem_ms = [
    begin 
      if(i < l+1)
        j+1
      elseif(i > nc-l)
        (ndofs-((npatch-1)*(p+1)))*(nc-(npatch-1))+(j)-(ndofs-1-((npatch-1)*(p+1)))
      else
        (ndofs-(2l*(p+1)))*(i-l)+j-(ndofs-1-(2l*(p+1)))
      end
    end  
    for i=1:nc,j=0:ndofs-1]
    local assem_MS_MS = ([elem_ms[t,ti] for  t=1:nc, ti=1:ndofs, _=1:ndofs], 
                         [elem_ms[t,tj] for  t=1:nc, _=1:ndofs, tj=1:ndofs], 
                         [elem_ms[t,ti] for t=1:nc, ti=1:ndofs])
    local sKms = zeros(Float64,nc,ndofs,ndofs)
    local sFms = zeros(Float64,nc,ndofs)
    local local_basis_vecs = zeros(Float64, q*nf+1, ndofs)

    # Compute the local and global stiffness matrices
    local cache1 = Kϵ, MS_Basis, local_basis_vecs, deepcopy(local_basis_vecs), similar(sKms[1,:,:])
    fillsKms!(sKms, cache1, nc, p, l)
    local cache2 = Fϵ, MS_Basis, local_basis_vecs, similar(sFms[1,:])
    fillsFms!(sFms, cache2, nc, p, l)

    # Assemble and the global system    
    local Kₘₛ = sparse(vec(assem_MS_MS[1]), vec(assem_MS_MS[2]), vec(sKms))
    local Fₘₛ = collect(sparsevec(vec(assem_MS_MS[3]), vec(sFms)))
    local sol = (Kₘₛ\Fₘₛ)
    local uhsol = zeros(Float64,nf+1)
    for j=1:nc, i=0:p
      uhsol[:] += sol[(p+1)*j+i-p]*view(MS_Basis,:,j,i+1)
    end
    # Compute the error in the solution
    error_cache = L²Error[itr], H¹Error[itr], basis_cache(q)
    L²Error[itr], H¹Error[itr] = error!(error_cache, u, ∇u, uhsol, nds_fine, elem_fine, quad)

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