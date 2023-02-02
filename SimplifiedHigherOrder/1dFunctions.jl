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
domain = (0.0,1.0)

#=
FEM parameters
=#
nc = 2^3 # Number of elements in the coarse space
nf = 2^11 # Number of elements in the fine space
p = 1 # Degree of polynomials in the coarse space
q = 1 # Degree of polynomials in the fine space
l = 10
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
elem_coarse = [i+j for i=1:nc, j=0:1]
elem_fine = [i+j for i=1:nf, j=0:1]
    
#=
Solve the saddle point problems to obtain the new basis functions
=#
# Store only the non-zero entries of the matrices of the saddle point problems
sKe = zeros(Float64,q+1,q+1,nf)
# Store the data for solving the multiscale problems
KDTrees = Vector{KDTree}(undef,nc)
Basis = Array{Float64}(undef,q*nf+1,nc,p+1)
fill!(Basis,0.0)
cache_q = basis_cache(q)
cache_p = Vector{Float64}(undef,p+1)
cache = sKe, Basis, cache_q, cache_p, KDTrees, q, p

compute_ms_basis!(cache, domain, nds, elem_coarse, elem_fine, D, l)

plt1 = plot()
bc = basis_cache(elem_fine, q)
for el=1:nc
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

# Compute the local and global stiffness matrices
nds_fine = LinRange(domain[1], domain[2], nf+1)
sFeϵ = zeros(Float64,q+1,nf)
fillLoadVec!(sFeϵ, basis_cache(q), nds_fine, elem_fine, q, quad, f)
sKeϵ = zeros(Float64,q+1,q+1,nf)
fillsKe!(sKeϵ, basis_cache(q), nds_fine, elem_fine, q, quad)
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKeϵ))
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFeϵ)))

cache1 = Kϵ, Basis, local_basis_vecs, deepcopy(local_basis_vecs), similar(sKms[1,:,:])
fillsKms!(sKms, cache1, nc, p, l)
cache2 = Fϵ, Basis, local_basis_vecs, similar(sFms[1,:])
fillsFms!(sFms, cache2, nc, p, l)

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

