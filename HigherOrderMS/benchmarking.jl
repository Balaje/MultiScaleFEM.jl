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
nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch
sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
sKms, sFms = multiscale
bc = basis_cache(q)

## First solve the problem using the Direct method
#=
sKe_ϵ = zeros(Float64, q+1, q+1, nf)
sFe_ϵ = zeros(Float64, q+1, nf)
fillsKe!(sKe_ϵ, bc, nds_fine, elem_fine, q, quad, D₂) 
Kϵ = sparse(vec(assem_H¹H¹[1]), vec(assem_H¹H¹[2]), vec(sKe_ϵ))
fillLoadVec!(sFe_ϵ, bc, nds_fine, elem_fine, q, quad, f)
# ---- @btime fillLoadVec!($sFe_ϵ, $bc, $nds_fine, $elem_fine, $q, $quad, $f)
# 7.774 ms (0 allocations: 0 bytes)
Fϵ = collect(sparsevec(vec(assem_H¹H¹[3]), vec(sFe_ϵ)))
=#
cache = assembler_cache(nds_fine, elem_fine, quad, q)
fillsKe!(cache, D₂)
Kϵ = sparse(cache[5][1], cache[5][2], cache[5][3])
fillsFe!(cache,f)
# ---- @btime fillsFe!($cache,$f)
# 1.360 ms (24 allocations: 6.00 MiB)
Fϵ = collect(sparsevec(cache[6][1],cache[6][2]))
solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]

## Benchmark the method after running it 100 times (Eg. Simulating time-dependent problems)
cache = assembler_cache(nds_fine, elem_fine, quad, q)
Td = @belapsed begin 
  fillsKe!($cache, D₂)
  Kϵ = sparse($cache[5][1], $cache[5][2], $cache[5][3])
  cache = assembler_cache(nds_fine, elem_fine, quad, q)
  for times=1:10^3
    fillsFe!(cache,f)
    Fϵ = collect(sparsevec(cache[6][1],cache[6][2]))
    solϵ = Kϵ[2:q*nf,2:q*nf]\Fϵ[2:q*nf]
  end
end
#=
###### 31.712 s (122144 allocations: 40.58 GiB) ######
=#


#=
MULTISCALE METHOD
=#
## Now compute the multiscale basis. 
cache = bc, zeros(Float64,p+1), quad, preallocated_data
compute_ms_basis!(cache, nc, q, p, D₂) 
# This has to be done only once, so benchmarking it once
Tbasis = @belapsed compute_ms_basis!(cache, nc, q, p, D₂) 
#=
###### 684.982 ms (2194 allocations: 1.31 GiB) ######
=#
Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
Fₘₛ = zeros(Float64,nc*(p+1))
uhsol = zeros(Float64,nf+1)
sol_cache = similar(uhsol)
# Get the patch-wise stiffness and load contribution
contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
matrix_cache = mat_contribs!(contrib_cache, D₂)
vector_cache = vec_contribs!(contrib_cache, f)
cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
fillsKms!(sKms, cache, nc, p, l)
cache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
fillsFms!(sFms, cache, nc, p, l)
cache3 = Kₘₛ, Fₘₛ
assemble_MS!(cache3, sKms, sFms, ms_elem)
Kₘₛ, Fₘₛ = cache3
sol = Kₘₛ\Fₘₛ
cache2 = uhsol, sol_cache
build_solution!(cache2, sol, local_basis_vecs)

# Now time solving the problem using the MS-Method
contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
Tms = @belapsed begin
  matrix_cache = mat_contribs!(contrib_cache, D₂)
  cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
  fillsKms!(sKms, cache, nc, p, l)
  for times=1:10^3
    vector_cache = vec_contribs!(contrib_cache, f)
    cache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
    fillsFms!(sFms, cache, nc, p, l)
    assemble_MS!(cache3, sKms, sFms, ms_elem)
    Kₘₛ, Fₘₛ = cache3
    sol = Kₘₛ\Fₘₛ
    build_solution!(cache2, sol, local_basis_vecs)
  end
end
#=
###### 15.388 s (190421 allocations: 11.26 GiB) ######
=#

# % GAIN using the MS method
gain = (Td - (Tbasis+Tms))/Td*100

#=
Check the new implementation with the old once
=#
assem_cache = assembler_cache(nds_fine, elem_fine, quad, q)
contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices);
# Check stiffness matrix contribution
@btime begin
  fillsKe!(assem_cache,D₁)
  Kϵ = sparse(assem_cache[5][1], assem_cache[5][2], assem_cache[5][3])
  matrix_cache = split_stiffness_matrix(Kϵ, elem_indices_to_global_indices)
end;
#=
 8.621 ms (170 allocations: 32.01 MiB)
=#
@btime begin
  mat_contribs!(contrib_cache, D₁)
end;
#=
 5.831 ms (420 allocations: 28.51 MiB)
=#

# Check load-vector contribution
@btime begin
  fillsFe!(assem_cache, f);
  Fϵ = collect(sparsevec(assem_cache[6][1],assem_cache[6][2]));
  vector_cache = split_load_vector(Fϵ, elem_indices_to_global_indices);
end;
#=
 15.502 ms (66 allocations: 12.00 MiB)
=#
@btime begin
  vec_contribs!(contrib_cache, f);
end;
#=
 10.368 ms (184 allocations: 11.50 MiB)
=#
#=
Conclusion: The new method of constructing the contributions is effective, especially when the aspect ratio (H/h) is large
=#