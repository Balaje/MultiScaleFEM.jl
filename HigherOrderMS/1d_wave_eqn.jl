##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### Julia program to solve a time-dependent problem #####
##### ##### ##### ##### ##### ##### ##### ##### ##### #####
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
include("time_dependent.jl")
 
# Problem data
# uₜₜ - (c(x)*uₓ)ₓ = f 
domain = (0.0,1.0)
c(x) = 4.0
#c(x) = (4.0 + cos(2π*x/(2e-2)))
f(x,t) = 0.0
U₀(x) = 0.0
U₁(x) = 4π*sin(2π*x)
Uₑ(x,t) = sin(2π*x)*sin(4π*t)

# Define the necessary parameters
nc = 2^1
nf = 2^11
p = 1
q = 1
l = 4
quad = gausslegendre(4)
Δt = 1e-5
tf = 0.125
@show tf
ntime = ceil(Int,tf/Δt)
plt = plot()

# Preallocate all the necessary data
preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
nds_coarse, elems_coarse, nds_fine, elem_fine = fullspace[1:4]
patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch[3:7]
ms_elem = assems[3]
sKms, sFms = multiscale
bc = basis_cache(q)


#### Solve the problem using the multiscale method ####
function fₙ_MS!(cache, tₙ::Float64)
  contrib_cache, Fms = cache
  vector_cache = vec_contribs!(contrib_cache, y->f(y,tₙ))
  fcache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
  fillsFms!(sFms, fcache, nc, p, l)
  assemble_MS_vector!(Fms, sFms, ms_elem)
  Fms
end
# Compute the Multiscale basis
cache = bc, zeros(Float64,p+1), quad, preallocated_data
compute_ms_basis!(cache, nc, q, p, c)
contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
matrix_cache = mat_contribs!(contrib_cache, c)
cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
fillsKms!(sKms, cache, nc, p, l)
## = The mass matrix
sMms = similar(sKms)
for i=1:nc
  sMms[i] = zeros(Float64,size(sKms[i]))    
end
matrix_cache = mat_contribs!(contrib_cache, x->1.0; matFunc=fillsMe!)
cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
fillsKms!(sMms, cache, nc, p, l)
## =
## = Assemble the stiffness and mass matrices
Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
Mₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
assemble_MS_matrix!(Kₘₛ, sKms, ms_elem)
assemble_MS_matrix!(Mₘₛ, sMms, ms_elem)
## = Preallocate the RHS vector
print("Begin solving using Multiscale Method ... \n")
let
  Fₘₛ = zeros(Float64,nc*(p+1))
  cache = contrib_cache, Fₘₛ
  Uₙ = setup_initial_condition(U₀, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ)
  Vₙ = setup_initial_condition(U₁, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ)
  M⁺ = (Mₘₛ + Δt^2/4*Kₘₛ)
  M⁻ = (Mₘₛ - Δt^2/4*Kₘₛ)
  Fₙ = (fₙ_MS!(cache,0.0)+fₙ_MS!(cache,Δt))*0.5
  Uₙ₊₁ = M⁺\(M⁻*Uₙ + (Δt)*M⁺*Vₙ + (Δt)^2/4*Fₙ) 
  Uₙ₊₂ = similar(Uₙ)
  fill!(Uₙ₊₂, 0.0)
  t = Δt
  for i=2:ntime
    Uₙ₊₂ = CN!(cache, t, Uₙ, Uₙ₊₁, Δt, M⁺, M⁻, fₙ_MS!)  
    Uₙ = Uₙ₊₁
    Uₙ₊₁ = Uₙ₊₂
    (i%1000 == 0) && print("Done t="*string(t+Δt)*"\n")
    t += Δt
  end
  (isnan(sum(Uₙ₊₂))) && print("\nUnstable \n")
  uhsol = zeros(Float64,q*nf+1)
  sol_cache = similar(uhsol)
  cache2 = uhsol, sol_cache
  build_solution!(cache2, Uₙ₊₂, local_basis_vecs)
  uhsol = cache2[1]
  plot!(plt, nds_fine, uhsol, label="Approximate sol. (MS Method)", lw=2, lc=:red)
  # uexact = [Uₑ(x,tf) for x in nds_fine]   
  # plot!(plt, nds_fine, uexact, label="Exact solution", lw=2, lc=:green)
end
print("End solving using Multiscale Method.\n\n")