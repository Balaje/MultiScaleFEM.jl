######## ############# ############# ############# ############# ######
# Script to perform some benchmarking tests for the multiscale method #
######## ############# ############# ############# ############# ######

using Plots
using BenchmarkTools
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

include("basis_functions.jl")
include("assemble_matrices.jl")
include("preallocate_matrices.jl")

#=
Problem data 2: Oscillatory diffusion coefficient
=#
domain = (0.0,1.0)
D₂(x) = (2 + cos(2π*x/(2^-7)))^-1
f(x) = 0.5*π^2*sin(π*x)

## FEM parameters
nc = 2^2 # Number of elements in the coarse space
nf = 2^16 # Number of elements in the fine space
p = 1 # Degree of polynomials in the coarse space
q = 1 # Degree of polynomials in the fine space
l = 4
quad = gausslegendre(2)

## Some preallocated data
preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
nds_fineₛ, elem_fineₛ = fine
nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, global_to_patch_indices, L, Lᵀ, ipcache = patch
sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
sKms, sFms = multiscale
bc = basis_cache(q)

## First solve the problem using the Direct method
sKe_ϵ = zeros(Float64, q+1, q+1, nf)
sFe_ϵ = zeros(Float64, q+1, nf)
fillsKe!(sKe_ϵ, bc, nds_fine, elem_fine, q, quad, D₂) 
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
fillLoadVec!(sFe_ϵ, bc, nds_fine, elem_fine, q, quad, f)
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]

## Benchmark the method after running it 100 times (Eg. Simulating time-dependent problems)
@btime begin 
  fillsKe!(sKe_ϵ, bc, nds_fine, elem_fine, q, quad, D₂) 
  Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
  for times=1:10^3
    fillLoadVec!(sFe_ϵ, bc, nds_fine, elem_fine, q, quad, f)
    Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
    solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]
  end
end
#=
27.276 s (109001 allocations: 34.68 GiB)
=#

## Now compute the multiscale basis. 
cache = bc, zeros(Float64,p+1), quad, preallocated_data
compute_ms_basis!(cache, nc, q, p, D₂) 

# This has to be done once, so benchmarking it once
@btime compute_ms_basis!(cache, nc, q, p, D₂) 
#=
686.264 ms (2190 allocations: 1.31 GiB)
=#

# Now time solving the problem using the MS-Method
Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
Fₘₛ = zeros(Float64,nc*(p+1))
uhsol = zeros(Float64,nf+1)
sol_cache = similar(uhsol)
matrix_cache = split_stiffness_matrix(sKe_ϵ, (assem_H¹H¹[1],assem_H¹H¹[2]), global_to_patch_indices)
cache = local_basis_vecs, global_to_patch_indices, L, Lᵀ, matrix_cache, ipcache
fillsKms!(sKms, cache, nc, p, l)
vector_cache = split_load_vector(sFe_ϵ, assem_H¹H¹[3], global_to_patch_indices)
cache = local_basis_vecs, global_to_patch_indices, Lᵀ, vector_cache
fillsFms!(sFms, cache, nc, p, l)
cache3 = Kₘₛ, Fₘₛ
assemble_MS!(cache3, sKms, sFms, ms_elem)
Kₘₛ, Fₘₛ = cache3
sol = Kₘₛ\Fₘₛ
cache2 = uhsol, sol_cache
build_solution!(cache2, sol, local_basis_vecs)

@btime begin
  matrix_cache = split_stiffness_matrix(sKe_ϵ, (assem_H¹H¹[1],assem_H¹H¹[2]), global_to_patch_indices)
  cache = local_basis_vecs, global_to_patch_indices, L, Lᵀ, matrix_cache, ipcache
  fillsKms!(sKms, cache, nc, p, l)
  for times=1:10^3
    ###### Bottleneck ######
    fillLoadVec!(sFe_ϵ, bc, nds_fine, elem_fine, q, quad, f)
    vector_cache = split_load_vector(sFe_ϵ, assem_H¹H¹[3], global_to_patch_indices)
    ######
    cache = local_basis_vecs, global_to_patch_indices, Lᵀ, vector_cache
    fillsFms!(sFms, cache, nc, p, l)
    assemble_MS!(cache3, sKms, sFms, ms_elem)
    Kₘₛ, Fₘₛ = cache3
    sol = Kₘₛ\Fₘₛ
    build_solution!(cache2, sol, local_basis_vecs)
  end
end
#=
15.680 s (152235 allocations: 6.37 GiB)
=#