########### ########### ########### ########
## Rate of convergence for rough data
########### ########### ########### ########

using Plots
using BenchmarkTools
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature
using Random
using Test

include("basis_functions.jl")
include("assemble_matrices.jl")

#=
Problem data
=#
f(x) = sin(5π*x)
ϵ = 2^-8
domain = (0.0,1.0)
nds_microscale = domain[1]:ϵ:domain[2]
tree_microscale = KDTree(nds_microscale')
pt_to_ind = randperm(length(nds_microscale))
pt_to_rand = Dict(zip(nds_microscale, pt_to_ind))

function D(x; tree=tree_microscale::KDTree, D=pt_to_rand::Dict)
  idx, = knn(tree,[x],2,true)
  a,b = view(tree.data,idx)
  (first(view(a,1)) < first(view(b,1))) && 
  return ( (first(view(a,1)) ≤ x ≤ first(view(b,1))) ? 
  begin 
    Random.seed!(get(D,first(view(a,1)),1234))
    0.5 + (10-0.5)*rand()  
  end :  0)
  (first(view(b,1)) < first(view(a,1))) && return ( (first(view(b,1)) ≤ x ≤ first(view(a,1))) ? 
  begin 
    Random.seed!(get(D,first(view(b,1)),1234))
    0.5 + (10-0.5)*rand()  
  end :  0)
end

elem = [i+j for i=1:length(nds_microscale)-1, j=0:1]
nds_elem_microscale = nds_microscale[elem]
function D_2(x; nds=nds_elem_microscale::Matrix{Float64}, D=pt_to_rand::Dict)
  nel = size(nds,1)
  for t=1:nel
    if(first(view(nds,t,1)) ≤ x ≤ first(view(nds,t,2))) # Same as nds[elem[t,1]] ≤ x ≤ nds[elem[t,2]]
      Random.seed!(get(D,first(view(nds,t,1)),1234))
      return 0.5 + (10-0.5)*rand()
    else
      continue
    end
  end
  return 0
end


# First check if the functions return the same value
#= @testset begin
  for k=1:10
    pt = rand()  
    @test D(pt; tree=tree_microscale, D=pt_to_rand) ≈ D_2(pt; nds=nds_elem_microscale, D=pt_to_rand)
  end
end =#
# @show D(0.999, tree; D=pt_to_rand), D_2(0.999, nds_elem; D=pt_to_rand) 

# Now time the functions
# @btime begin for i=1:10^6 D(0.999; tree=$tree_microscale, D=$pt_to_rand) end end
# @btime begin for i=1:10^6 D_2(0.999; nds=$nds_elem_microscale, D=$pt_to_rand) end end

#=
Summary: For ϵ = 2^-12
@btime begin for i=1:10^6 D(0.999; tree=$tree_microscale, D=$pt_to_rand) end end
836.764 ms (10000000 allocations: 671.39 MiB)
@btime begin for i=1:10^6 D_2(0.999; nds=$nds_elem_microscale, D=$pt_to_rand) end end  
3.561 s (7000000 allocations: 457.76 MiB)

Memory allocations are due to Random.seed!() + knn().
- Linear search takes less memory but runs slower. [Only Random.seed!]
- KDTree search takes more memory but runs quicker. [Random.seed! + knn()]
=#

### - Let us use D() with KDTree ###
"""
Function to compute the l2 and energy errors
"""
function error(bc, ue::Function, ∇ue::Function, uh::AbstractVector{Float64}, nds::AbstractVector{Float64},
  elem::AbstractMatrix{Int64}, tree::KDTree, KDTrees::Vector{KDTree}, 
  quad::Tuple{Vector{Float64},Vector{Float64}}; Nfine=200)
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

q=1
nf=2^11
qorder=2
quad=gausslegendre(qorder)
## Solve the problem directly with h=2^-16
h = (domain[2]-domain[1])/nf
nds_fine = domain[1]:h:domain[2]
tree_fine = KDTree(nds_fine')
elem_fine = [i+j for i=1:nf, j=0:1]
assem_H¹H¹ = ([(q)*t+ti-(q-1) for _=0:q, ti=0:q, t=1:nf], 
[(q)*t+tj-(q-1) for tj=0:q, _=0:q, t=1:nf], 
[(q)*t+ti-(q-1) for ti=0:q, t=1:nf])  
sKe_ϵ = zeros(Float64,q+1,q+1,nf)
sFe_ϵ = zeros(Float64,q+1,nf)
cache_fine_scale = basis_cache(q)
fillsKe!(sKe_ϵ, cache_fine_scale, nds_fine, elem_fine, q, quad)
fillLoadVec!(sFe_ϵ, cache_fine_scale, nds_fine, elem_fine, q, quad, f)
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
tn = 1:lastindex(nds_fine)
bn = [1,lastindex(nds_fine)]
fn = setdiff(tn, bn)
solϵ = Kϵ[fn,fn]\Fϵ[fn]
solϵ = vcat(0.0,solϵ,0.0)
p1 = plot(nds_fine, solϵ, label="Fine scale solution")
sol_cache = basis_cache(elem_fine, q)
uₕ(x) = Λₖ(sol_cache, x, solϵ, tree_fine) # This serves as the exact solution for the rate of convergence
∇uₕ(x) = ∇Λₖ(sol_cache, x, solϵ, tree_fine) # This serves as the exact solution for the rate of convergence

# Now solve the problem using the MS method
𝒩 = [4,8,16,32,64]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))
p = 1

plt = plot()
plt1 = plot()
plt2 = plot()

for l in [4,5,6,7,8]
  for (nc,itr) in zip(𝒩,1:length(𝒩))
    #=
    Precompute all the caches. Essential for computing the solution quickly
    =#
    npatch = min(2l+1,nc)
    # Construct the coarse mesh
    H = (domain[2]-domain[1])/nc
    nds_coarse = domain[1]:H:domain[2]
    tree = KDTree(nds_coarse')
    elem_coarse = [i+j for i=1:nc, j=0:1]    
    assem_L²L² = ([(p+1)*t+ti-p for _=0:p, ti=0:p, t=1:nc], 
    [(p+1)*t+tj-p for tj=0:p, _=0:p, t=1:nc], 
    [(p+1)*t+ti-p for ti=0:p, t=1:nc])  
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
    cache = sKe, ssem_H¹H¹, MS_Basis, cache_q, cache_p, KDTrees, q, p
    compute_ms_basis!(cache, domain, nds_coarse, elem_coarse, elem_fine, D, l)
    
    #=
      Now solve the multiscale problems.
        The multiscale basis are stored in the variable MS_Basis
    =#
    # Compute the local and global stiffness matrices
    cache1 = Kϵ, MS_Basis, local_basis_vecs, deepcopy(local_basis_vecs), similar(sKms[1,:,:])
    fillsKms!(sKms, cache1, nc, p, l)
    cache2 = Fϵ, MS_Basis, local_basis_vecs, similar(sFms[1,:])
    fillsFms!(sFms, cache2, nc, p, l)
    # Assemble and the global system
    Kₘₛ = sparse(vec(assem_MS_MS[1]), vec(assem_MS_MS[2]), vec(sKms))
    Fₘₛ = collect(sparsevec(vec(assem_MS_MS[3]), vec(sFms)))
    sol = (Kₘₛ\Fₘₛ)

    cache = basis_cache(elem_coarse, elem_fine, p, q, l, MS_Basis)
    L²Error[itr], H¹Error[itr] = error(cache, uₕ, ∇uₕ, sol, nds_coarse, elem_coarse, tree, KDTrees, quad; Nfine=20)

    println("Done nc = "*string(nc))    
  end

  plot!(plt, 1 ./𝒩, L²Error, label="L² (l="*string(l)*")", lw=2)
  plot!(plt1, 1 ./𝒩, H¹Error, label="Energy (l="*string(l)*")", lw=2)
  scatter!(plt, 1 ./𝒩, L²Error, label="", markersize=2)
  scatter!(plt1, 1 ./𝒩, H¹Error, label="", markersize=2, legend=:best)
end

plot!(plt1, 1 ./𝒩, (1 ./𝒩).^3, label="Order 3", ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)
plot!(plt, 1 ./𝒩, (1 ./𝒩).^4, label="Order 4", ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)