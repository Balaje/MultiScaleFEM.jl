###### ######## ######## ######## ####
# Main file containing the functions #
###### ######## ######## ######## ####
using Plots
using BenchmarkTools
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

include("basis_functions.jl")
include("assemble_matrices.jl")

nc = 2^5 # Number of elements in the coarse space
nf = 2^10 # Number of elements in the fine space
p = 1 # Degree of polynomials in the coarse space
q = 1 # Degree of polynomials in the fine space
l = 3 # Size of patch
qorder = 2
quad = gausslegendre(qorder)

# Problem data
D(x) = @. 0.5
f(x) = @. 1.0

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
elem_coarse = [i+j for i=1:nc, j=0:1]
elem_fine = [i+j for i=1:nf, j=0:1]
    
#=
Solve the saddle point problems to obtain the new basis functions
=#
KDTrees = Vector{KDTree}(undef,nc)
Basis = Array{Float64}(undef,q*nf+1,nc,p+1)
for i=1:nc
  start = max(1,(i-l)) - (((i+l) > nc) ? abs(i+l-nc) : 0) # Start index of patch
  last = min(nc,(i+l)) + (((i-l) < 1) ? abs(i-l-1) : 0) # Last index of patch
  # Get the patch domain and connectivity
  patch_elem = elem_coarse[start:last,:] 
  patch = (nds[minimum(patch_elem)], nds[maximum(patch_elem)])  
  patch_elem = patch_elem .- (minimum(patch_elem) - 1)
  # Build the FEM nodes
  local h = (patch[2]-patch[1])/nf
  nds_fine = patch[1]:h:patch[2]
  nds_coarse = patch[1]:H:patch[2]
  # Build some data structures
  KDTrees[i] = KDTree(nds_fine') # KDTree for searching the points
  cache_q = basis_cache(q)
  cache_p = Vector{Float64}(undef,p+1)  
  # Fill up the matrices
  fillsKe!(sKe, cache_q, nds_fine, elem_fine, q, quad)
  fillsLe!(sLe, (cache_q,cache_p), nds_fine, nds_coarse, elem_fine, patch_elem, (q,p), quad)
  # Basis function of Vₕᵖ(K)
  function fₖ(x::Float64, j::Int64)
    res = Vector{Float64}(undef,p+1)
    nodes = [nds[elem_coarse[i,1]], nds[elem_coarse[i,2]]]
    Λₖ!(res, x, nodes, p)
    res[j]
  end
  # Compute new the basis functions
  for j=1:p+1
    # Fill up the vector
    fillsFe!(sFe, cache_p, nds_coarse, patch_elem, p, quad, y->fₖ(y,j))
    # Assemble the matrices
    KK = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe))
    LL = sparse(vec(assem_H¹L²[1]), vec(assem_H¹L²[2]), vec(sLe))
    FF = collect(sparsevec(vec(assem_H¹L²[3]), vec(sFe)))
    # Apply the boundary conditions
    tn = 1:q*nf+1
    bn = [1,q*nf+1]
    fn = setdiff(tn,bn)
    K = KK[fn,fn]; L = LL[fn,:]; F = FF
    # Solve the problem
    LHS = [K L; L' spzeros(Float64,size(L,2),size(L,2))]
    dropzeros!(LHS)
    RHS = vcat(zeros(Float64,size(K,1)),F);
    RHS = LHS\RHS
    Λ = vcat(0, RHS[1:size(K,1)], 0)
    copyto!(view(Basis,:,i,j), Λ)  
  end
end

## Plot one basis function
el = 2
bases = basis_cache(elem_fine, q)
xvals = [x[1] for x in KDTrees[el].data]
fxvals = [∇Λₖ(bases, x, Basis[:,el,2], KDTrees[el]) for x in xvals]
plt = plot(xvals,fxvals);
xlims!(plt,(0,1))

#=
Solve the MultiScale problem
=#
ndofs = ((2l+1) < nc) ? (2l+1)*(p+1) : nc
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
assem_MS_MS = ([elem_ms[t,ti] for  t=1:nc, _=1:ndofs, ti=1:ndofs], 
  [elem_ms[t,tj] for  t=1:nc, tj=1:ndofs, _=1:ndofs], 
  [elem_ms[t,ti] for t=1:nc, ti=1:ndofs])
tree = KDTree(nds')
sKms = zeros(Float64,ndofs,ndofs,nc)
sFms = zeros(Float64,ndofs,nc)

# Compute the local and global stiffness matrices
bases = basis_cache(elem_fine, q)
local_basis_vecs = zeros(Float64, q*nf+1, (2l+1)*(p+1))
cache = KDTrees, Basis, local_basis_vecs, bases
fillsKms!(sKms, cache, nds, elem_ms, p, l, quad; Nfine=2)
fillsFms!(sFms, cache, nds, elem_ms, p, l, quad, f; Nfine=2)

# Assemble and the global system
Kₘₛ = sparse(vec(assem_MS_MS[1]), vec(assem_MS_MS[2]), vec(sKms))
Fₘₛ = collect(sparsevec(vec(assem_MS_MS[3]), vec(sFms)))
sol = (Kₘₛ\Fₘₛ)

bases = basis_cache(elem_coarse, elem_fine, p, q, l, Basis)
# solnds = [uₘₛ(bases, x, sol, tree, KDTrees) for x in nds];
uₘₛ(bases, 0.1, sol, tree, KDTrees)
