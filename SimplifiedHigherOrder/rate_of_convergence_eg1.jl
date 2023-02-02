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
# ε = 2^-5
# D(x) = @. (2 + cos(2π*x/ε))^(-1)
# f(x) = @. 1.0
# u(x) = @. (x - x^2 + ε*(1/(4π)*sin(2π*x/ε) - 1/(2π)*x*sin(2π*x/ε) - ε/(4π^2)*cos(2π*x/ε) + ε/(4π^2)))
domain = (0.0,1.0)

"""
Function to compute the l2 and energy errors
"""
function error(bc, ue::Function, uh::AbstractVector{Float64}, nds::AbstractVector{Float64},
  elem::AbstractMatrix{Int64}, tree::KDTree, KDTrees::Vector{KDTree}, 
  quad::Tuple{Vector{Float64},Vector{Float64}}; Nfine=200)
  ∇ue(x) = ForwardDiff.derivative(ue,x)
  qs, ws = quad
  l2err = 0.0; h1err = 0.0
  nc = size(elem,1)
  for t=1:nc, i=1:lastindex(qs)
    cs = view(nds,view(elem,t,:))
    hlocal = (cs[2]-cs[1])/Nfine
    xlocal = cs[1]:hlocal:cs[2]
    for j=1:lastindex(xlocal)-1
      x̂ = (xlocal[j+1]-xlocal[j])*0.5 + (xlocal[j+1]-xlocal[j])*0.5*qs[i]
      l2err += ws[i]*(ue(x̂) - uₘₛ(bc, x̂, uh, tree, KDTrees))^2*(hlocal)*0.5
      h1err += ws[i]*D(x̂)*(∇ue(x̂) - ∇uₘₛ(bc, x̂, uh, tree, KDTrees))^2*(hlocal)*0.5
    end
  end
  sqrt(l2err), sqrt(h1err)
end

#=
Constant paramters
=#
p = 1; q = 1;
nf = 2^11
qorder = 2
quad = gausslegendre(qorder)

𝒩 = [2,4,8,16,32,64]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

plt = plot()
plt1 = plot()
plt2 = plot()

for l in [4,5,6]
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    #=
    Precompute all the caches. Essential for computing the solution quickly
    =#
    npatch = min(2l+1,nc)
    # Construct the coarse mesh
    H = (domain[2]-domain[1])/nc
    nds_coarse = domain[1]:H:domain[2]
    tree = KDTree(nds_coarse')
    assem_L²L² = ([(p+1)*t+ti-p for _=0:p, ti=0:p, t=1:nc], 
    [(p+1)*t+tj-p for tj=0:p, _=0:p, t=1:nc], 
    [(p+1)*t+ti-p for ti=0:p, t=1:nc])
    assem_H¹H¹ = ([(q)*t+ti-(q-1) for _=0:q, ti=0:q, t=1:nf], 
    [(q)*t+tj-(q-1) for tj=0:q, _=0:q, t=1:nf], 
    [(q)*t+ti-(q-1) for ti=0:q, t=1:nf])           
    # Connectivity of the coarse and fine domains
    elem_coarse = [i+j for i=1:nc, j=0:1]
    elem_fine = [i+j for i=1:nf, j=0:1]
    # Store only the non-zero entries of the matrices of the saddle point problems
    sKe = zeros(Float64,q+1,q+1,nf)
    # Store the data for solving the multiscale problems
    KDTrees = Vector{KDTree}(undef,nc)
    MS_Basis = Array{Float64}(undef,q*nf+1,nc,p+1)
    # Some precomputed/preallocated data for the multiscale problem
    ndofs = npatch*(p+1)
    elem_ms = [
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
    assem_MS_MS = ([elem_ms[t,ti] for  t=1:nc, ti=1:ndofs, _=1:ndofs], 
    [elem_ms[t,tj] for  t=1:nc, _=1:ndofs, tj=1:ndofs], 
    [elem_ms[t,ti] for t=1:nc, ti=1:ndofs])
    sKms = zeros(Float64,nc,ndofs,ndofs)
    sFms = zeros(Float64,nc,ndofs)
    local_basis_vecs = zeros(Float64, q*nf+1, ndofs)
        
    #=
    Efficiently compute the solution to the saddle point problems.
    =#
    cache_q = basis_cache(q)
    cache_p = Vector{Float64}(undef,p+1)
    cache = sKe, assem_H¹H¹, MS_Basis, cache_q, cache_p, KDTrees, q, p
    compute_ms_basis!(cache, domain, nds_coarse, elem_coarse, elem_fine, D, l)
    
    #=
      Now solve the multiscale problems.
        The multiscale basis are stored in the variable MS_Basis
    =#
    # Compute the local and global stiffness matrices
    sKeϵ = zeros(Float64,q+1,q+1,nf)
    sFeϵ = zeros(Float64,q+1,nf)
    nds_fine = LinRange(domain[1],domain[2],nf+1)    
    fillsKe!(sKeϵ, basis_cache(q), nds_fine, elem_fine, q, quad)
    fillLoadVec!(sFeϵ, basis_cache(q), nds_fine, elem_fine, q, quad, f)
    Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKeϵ))
    Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFeϵ)))
    cache1 = Kϵ, MS_Basis, local_basis_vecs, deepcopy(local_basis_vecs), similar(sKms[1,:,:])
    fillsKms!(sKms, cache1, nc, p, l)
    cache2 = Fϵ, MS_Basis, local_basis_vecs, similar(sFms[1,:])
    fillsFms!(sFms, cache2, nc, p, l)

    # Assemble and the global system
    Kₘₛ = sparse(vec(assem_MS_MS[1]), vec(assem_MS_MS[2]), vec(sKms))
    Fₘₛ = collect(sparsevec(vec(assem_MS_MS[3]), vec(sFms)))
    sol = (Kₘₛ\Fₘₛ)
    # Compute the error in the solution
    bc = basis_cache(elem_coarse, elem_fine, p, q, l, MS_Basis)
    L²Error[itr], H¹Error[itr] = error(bc, u, sol, nds_coarse, elem_coarse, tree, KDTrees, quad; Nfine=nf)

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