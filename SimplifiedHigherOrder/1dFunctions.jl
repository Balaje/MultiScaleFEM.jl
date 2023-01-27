###### ######## ######## ######## ####
# Main file containing the functions #
###### ######## ######## ######## ####
using StaticArrays
using Plots
using BenchmarkTools
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

include("basis_functions.jl")

const nc = 4 # Number of elements in the coarse space
const nf = 200 # Number of elements in the fine space
const p = 4 # Degree of polynomials in the coarse space
const q = 1 # Degree of polynomials in the fine space
const l = 2 # Size of patch
const ndofs = ((2l+1) < nc) ? (2l+1)*(p+1) : nc
const qorder = 2
quad = gausslegendre(qorder)

# Problem data
D(x) = @. 1.0

# Construct the coarse mesh
domain = (0,1)
H = (domain[2]-domain[1])/nc
nds = domain[1]:H:domain[2]
tree = KDTree(nds')
# Element conectivity for finite elements
l²elem = @SMatrix[(p+1)*i+j-p for i=1:nc, j=0:p]
h¹elem = @SMatrix[(q)*i+j-(q-1) for i=1:nf, j=0:q]
ms_elem = @SMatrix [
  begin 
    if(i < l+1)
      j+1
    elseif(i > nc-l)
      (ndofs-(2l*(p+1)))*(nc-2l)+j-(ndofs-1-(2l*(p+1)))
    else
      (ndofs-(2l*(p+1)))*(i-l)+j-(ndofs-1-(2l*(p+1)))
    end
  end  
  for i=1:nc,j=0:ndofs-1]

# Get the local-global correspondence for the inner project
assem_L²L² = (@SArray[(p+1)*t+ti-p for _=0:p, ti=0:p, t=1:nc], 
@SArray[(p+1)*t+tj-p for tj=0:p, _=0:p, t=1:nc], 
@SMatrix[(p+1)*t+ti-p for ti=0:p, t=1:nc])

assem_H¹H¹ = (@SArray[(q)*t+ti-(q-1) for _=0:q, ti=0:q, t=1:nf], 
@SArray[(q)*t+tj-(q-1) for tj=0:q, _=0:q, t=1:nf], 
@SMatrix[(q)*t+ti-(q-1) for ti=0:q, t=1:nf])    

assem_H¹L² = (@SArray[(q)*Q+qᵢ-(q-1) for Q=1:nf, P=1:nc, qᵢ=0:q, _=0:p], 
@SArray[(p+1)*P+pᵢ-p for Q=1:nf, P=1:nc, _=0:q, pᵢ=0:p], 
@SMatrix[(p+1)*P+pᵢ-p for pᵢ=0:p, P=1:nc])                          
# Λₖ!(res, 0.0, [-1.0,1.0], p)
# 
#=
Solve the saddle point problem
1) First compute the basis of the L² space
=#

patch = (0,0.1)
h = (patch[2]-patch[1])/nf
nds_patch = patch[1]:h:patch[2]
cache = basis_cache(q)

# Store only the non-zero entries of the stiffness matrix
sKe = @MArray [0.0 for _=0:q, _=0:q, _=1:nf]

# sKe = Array{Float64}(undef, size(assem_H¹H¹[1]))
# fill!(sKe,0.0)

function fillsKe!(sKe::T1, cache, nds_patch::AbstractVector{Float64},
  h¹elem::T2, quad::Tuple{Vector{Float64}, Vector{Float64}}) where {T1<:MArray, T2<:SMatrix}

  nf = size(h¹elem,1)
  for t=1:nf
    cs = view(nds_patch, view(h¹elem, t, :))
    hlocal = cs[2]-cs[1]
    qs,ws = quad
    for i=1:lastindex(qs)
      x̂ = (cs[2]+cs[1])*0.5 + (cs[2]-cs[1])*0.5*qs[i]
      basis = ∇ϕᵢ!(cache, x̂)
      for j=1:q+1, k=1:q+1
        sKe[j,k,t] += ws[i]*D(x̂)*basis[j]*basis[k]*(hlocal*0.5)^-1
      end
    end
  end
end

fill!(sKe,0.0)
fillsKe!(sKe, cache, nds_patch, h¹elem, quad);
# @btime fillsKe!($sKe, $cache, $nds_patch, $h¹elem, $quad)
K = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe))
# # Assemble the stiffness matrix 
# local_stiffness = ones(Float64,q+1,q+1)
# sA = Vector{Float64}(undef, (q+1)^2*nf)
# fill!(sA,0.0)
# index=0
# for i=1:q+1,j=1:q+1
#   sA[index+1:index+nf] .= local_stiffness[i,j]
#   global index += nf
# end
# K = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), sA)

# local_stiffness = ones(Float64,q+1,p+1)
# sA = Array{Float64}(undef, size(assem_H¹L²[1]))
# fill!(sA,0.0)
# for tf=1:nf, tc=1:nc, i=1:q+1, j=1:p+1
#   sA[tf,tc,i,j] = local_stiffness[i,j]
# end
# L = sparse(vec(assem_H¹L²[1]), vec(assem_H¹L²[2]), vec(sA))