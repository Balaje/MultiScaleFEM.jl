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

#=
Problem parameters
=#
# Problem data
D(x) = @. 0.5
f(x) = @. 1.0
u(x) = @. x*(1-x)
domain = (0,1)

#=
FEM parameters
=#
nc = 2^3 # Number of elements in the coarse space
nf = 2^9 # Number of elements in the fine space
p = 1 # Degree of polynomials in the coarse space
q = 1 # Degree of polynomials in the fine space
l = 6
npatch = min(2l+1,nc) # Number of elements in patch
@show l
qorder = 2
quad = gausslegendre(qorder)
# Construct the coarse mesh
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
assem_H¹L² = ([(q)*Q+qᵢ-(q-1) for qᵢ=0:q, _=0:p, P=1:npatch, Q=1:nf], 
  [(p+1)*P+pᵢ-p for _=0:q, pᵢ=0:p, P=1:npatch, Q=1:nf], 
  [(p+1)*P+pᵢ-p for pᵢ=0:p, P=1:npatch])    
# Connectivity of the fine elements
elem_coarse = [i+j for i=1:nc, j=0:1]
elem_fine = [i+j for i=1:nf, j=0:1]
    
#=
Solve the saddle point problems to obtain the new basis functions
=#
# Store only the non-zero entries of the matrices of the saddle point problems
sKe = zeros(Float64,q+1,q+1,nf)
sLe = zeros(Float64,q+1,p+1,npatch,nf)
sFe = zeros(Float64,p+1,npatch)
# Store the data for solving the multiscale problems
KDTrees = Vector{KDTree}(undef,nc)
Basis = Array{Float64}(undef,q*nf+1,nc,p+1)
for i=1:nc
  start = max(1,i-l)
  last = min(nc,i+l)
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
    zcols = size(LHS,1)-rank(LHS)
    LHS = LHS[1:end-zcols,1:end-zcols]    
    # display(LHS[264,:])
    dropzeros!(LHS)
    RHS = vcat(zeros(Float64,size(K,1)),F);
    RHS = RHS[1:end-zcols]
    RHS = LHS\RHS
    Λ = vcat(0, RHS[1:size(K,1)], 0)
    copyto!(view(Basis,:,i,j), Λ)  
  end
end

plt1 = plot()
bc = basis_cache(elem_fine, q)
for el=[1,nc]
  for ii=1:1
    xvals1 = 0:(1/1000):1
    fxvals1 = [Λₖ(bc, x, Basis[:,el,ii], KDTrees[el]) for x in xvals1]
    plot!(plt1, xvals1, fxvals1, xlims=(0,1))
  end
end

#=
Solve the MultiScale problem
=#
# Some precomputed/preallocated data
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
tree = KDTree(nds')
sKms = zeros(Float64,nc,ndofs,ndofs)
sFms = zeros(Float64,nc,ndofs)
bc = basis_cache(elem_fine, q)
local_basis_vecs = zeros(Float64, q*nf+1, ndofs)
binds_1 = zeros(Int64, ndofs)
cache = KDTrees, Basis, local_basis_vecs, bc, binds_1

# Compute the local and global stiffness matrices
fillsKms!(sKms, cache, elem_coarse, nds, p, l, quad; Nfine=nf)
fillsFms!(sFms, cache, elem_coarse, nds, p, l, quad, f; Nfine=nf)

# Assemble and the global system
Kₘₛ = sparse(vec(assem_MS_MS[1]), vec(assem_MS_MS[2]), vec(sKms))
dropzeros!(Kₘₛ)
Fₘₛ = collect(sparsevec(vec(assem_MS_MS[3]), vec(sFms)))
sol = (Kₘₛ\Fₘₛ)

# Compute the solution
bc = basis_cache(elem_coarse, elem_fine, p, q, l, Basis)
xvals = nds
uhxvals = [uₘₛ(bc, x, sol, tree, KDTrees) for x in xvals];
uxvals = u.(xvals)
plt = plot(xvals, uhxvals, lw=2, color=:red, label="Approx. Solution")
plot!(plt, xvals, uxvals, color=:black, label="Exact solution")

