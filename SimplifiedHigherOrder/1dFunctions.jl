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
include("assemble_matrices.jl")

const nc = 2^3 # Number of elements in the coarse space
const nf = 2^10 # Number of elements in the fine space
const p = 1 # Degree of polynomials in the coarse space
const q = 1 # Degree of polynomials in the fine space
const l = 1 # Size of patch
const ndofs = ((2l+1) < nc) ? (2l+1)*(p+1) : nc
const qorder = 2
quad = gausslegendre(qorder)

# Problem data
D(x) = @. 1.0

# Construct the coarse mesh
domain = (0,1)
H = (domain[2]-domain[1])/nc
nds = domain[1]:H:domain[2]

#=
Pre-compute the assemblers and the connectivity matrices
=#
assem_L²L² = ([(p+1)*t+ti-p for _=0:p, ti=0:p, t=1:nc], 
  [(p+1)*t+tj-p for tj=0:p, _=0:p, t=1:nc], 
  [(p+1)*t+ti-p for ti=0:p, t=1:nc])
assem_H¹H¹ = ([(q)*t+ti-(q-1) for _=0:q, ti=0:q, t=1:nf], 
  [(q)*t+tj-(q-1) for tj=0:q, _=0:q, t=1:nf], 
  [(q)*t+ti-(q-1) for ti=0:q, t=1:nf])          
assem_H¹L² = ([(q)*Q+qᵢ-(q-1) for qᵢ=0:q, _=0:p, P=1:2l+1, Q=1:nf], 
  [(p+1)*P+pᵢ-p for _=0:q, pᵢ=0:p, P=1:2l+1, Q=1:nf], 
  [(p+1)*P+pᵢ-p for pᵢ=0:p, P=1:2l+1])    
# Store only the non-zero entries of the stiffness matrix
sKe = zeros(Float64,q+1,q+1,nf)
sLe = zeros(Float64,q+1,p+1,2l+1,nf)
sFe = zeros(Float64,p+1,2l+1)
# Connectivity of the fine elements
elem_coarse = @SMatrix[i+j for i=1:nc, j=0:1]
elem_fine = @SMatrix[i+j for i=1:nf, j=0:1]
elem_ms = [
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
    
#=
Solve the saddle point problem
1) First compute the basis of the L² space
=#

for i=1:1
  patch_elem = elem_coarse[2:4,:] .- (minimum(elem_coarse[2:4,:]) - 1)
  patch = (nds[minimum(patch_elem)], nds[maximum(patch_elem)])
  h = (patch[2]-patch[1])/nf
  nds_fine = patch[1]:h:patch[2]
  nds_coarse = patch[1]:H:patch[2]
  global f(x) = @. (nds_coarse[patch_elem[2,1]] ≤ x ≤ nds_coarse[patch_elem[2,2]]) ? 1.0 : 0.0

  # Build some data structures
  # tree_fine = KDTree(nds_fine')
  cache_q = basis_cache(q)
  cache_p = Vector{Float64}(undef,p+1)  

  # Fill up the matrices
  fillsKe!(sKe, cache_q, nds_fine, elem_fine, q, quad)
  fillsLe!(sLe, (cache_q,cache_p), nds_fine, nds_coarse, elem_fine, patch_elem, (q,p), quad)
  fillsFe!(sFe, cache_p, nds_coarse, patch_elem, p, quad)

  # Assemble the matrices
  KK = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe))
  LL = sparse(vec(assem_H¹L²[1]), vec(assem_H¹L²[2]), vec(sLe))
  FF = collect(sparsevec(vec(assem_H¹L²[3]), vec(sFe)))

  # Apply the boundary conditions
  tn = 1:length(nds_fine)
  bn = [1,length(nds_fine)]
  fn = setdiff(tn,bn)
  K = KK[fn,fn]; L = LL[fn,:]; F = FF

  # Solve the problem
  LHS = [K L; L' spzeros(Float64,size(L,2),size(L,2))]
  dropzeros!(LHS)
  RHS = vcat(zeros(Float64,size(K,1)),F);
  RHS = LHS\RHS

  plt = plot(nds_fine[2:end-1], RHS[1:size(K,1)])
  xlims!(plt,(0,1))
  display(plt)
end