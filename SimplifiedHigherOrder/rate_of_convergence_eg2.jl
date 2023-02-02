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
ϵ = 2^-12
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
q=1
nf=2^16
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

# Now solve the problem using the MS method
𝒩 = [2]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))
l = [4]
p = 1

for l in [4]
  for (nc,i) in zip(𝒩,1:length(𝒩))
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
    bc = basis_cache(elem_fine, q)
    local_basis_vecs = zeros(Float64, q*nf+1, ndofs)
    cache = KDTrees, MS_Basis, local_basis_vecs, bc
    
    #=
    Efficiently compute the solution to the saddle point problems.
    =#
    cache_q = basis_cache(q)
    cache_p = Vector{Float64}(undef,p+1)
    cache = sKe, MS_Basis, cache_q, cache_p, KDTrees, q, p
    compute_ms_basis!(cache, domain, nds_coarse, elem_coarse, elem_fine, D, l)
    
    
  end
end