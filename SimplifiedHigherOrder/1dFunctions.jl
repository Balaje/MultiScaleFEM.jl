###### ######## ######## ######## ####
# Main file containing the functions #
###### ######## ######## ######## ####
using StaticArrays
using Plots
using BenchmarkTools
using NearestNeighbors
using SparseArrays

const nc = 4 # Number of elements in the coarse space
const nf = 8 # Number of elements in the fine space
const p = 3 # Degree of polynomials in the coarse space
const q = 1 # Degree of polynomials in the fine space
const l = 2 # Size of patch
const ndofs = ((2l+1) < nc) ? (2l+1)*(p+1) : nc

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

assem_H¹L² = (@SArray[(q)*Q+qᵢ-(q-1) for qᵢ=0:q, pᵢ=0:p, Q=1:nf, P=1:nc], 
@SArray[(p+1)*P+pᵢ-p for qᵢ=0:q, pᵢ=0:p, Q=1:nf, P=1:nc], 
@SMatrix[(p+1)*P+pᵢ-p for pᵢ=0:p, P=1:nc])                          

local_stiffness = ones(Float64,q+1,q+1)
sA = Vector{Float64}(undef, (q+1)^2*nf)
index=0
for i=1:q+1,j=1:q+1
  sA[index+1:index+nf] .= local_stiffness[i,j]
  global index += nf
end
K = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), sA)