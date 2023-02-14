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

#=
Problem data 2: Oscillatory diffusion coefficient
=#
domain = (0.0,1.0)
A(x) = (2 + cos(2π*x/(2e-2)))^-1
f(x,t) = 0.0
U₀(x) = sin(π*x)
Uₑ(x,t) = exp(-π^2*t)*U₀(x)

# Define the necessary parameters
nc = 2^1
nf = 2^11
p = 1
q = 1
l = 4
quad = gausslegendre(4)

# Preallocate all the necessary data
preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p));
fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
nds_fineₛ, elem_fineₛ = fine
nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch
sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
sKms, sFms = multiscale
bc = basis_cache(q)

# First obtain the stiffness and mass matrix in the fine scale
cache = assembler_cache(nds_fine, elem_fine, quad, q)
fillsKe!(cache, A)
Kϵ = sparse(cache[5][1], cache[5][2], cache[5][3])
fillsMe!(cache, x->1.0)
Mϵ = sparse(cache[5][1], cache[5][2], cache[5][3])
# The RHS-vector as a function of t
function fₙ!(fcache, tₙ::Float64)  
  cache, fn = fcache
  fillsFe!(cache, y->f(y,tₙ))
  F = collect(sparsevec(cache[6][1], cache[6][2]))
  F[fn]
end
# Solve the time-dependent problem using the direct method
Δt = 1e-4
tf = 1.0
ntime = ceil(Int,tf/Δt)
plt₁ = plot()
plt₂ = plot()
plt₃ = plot()
fn = 2:q*nf

print("Begin solving using Direct Method ... \n")
let 
  Uₙ = U₀.(nds_fine[fn])
  Uₙ₊₁ = similar(Uₙ)
  fill!(Uₙ₊₁, 0.0)
  t = 0
  cache = assembler_cache(nds_fine, elem_fine, quad, q), fn
  for i=1:ntime
    Uₙ₊₁ = RK4!(cache, t+Δt, Uₙ, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!)  
    Uₙ = Uₙ₊₁
    t += Δt
  end
  Uₙ₊₁ = vcat(0, Uₙ₊₁, 0)
  (isnan(sum(Uₙ₊₁))) && print("\nUnstable \n")
  plot!(plt₁, nds_fine, Uₙ₊₁, label="Approximate sol. (direct method)", lc=:black, lw=2)
end
print("End solving using Direct Method.\n\n")

#=
Begin solving the problem using the multiscale method
=#
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
compute_ms_basis!(cache, nc, q, p, A)
contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
matrix_cache = mat_contribs!(contrib_cache, A)
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
  Uₙ₊₁ = similar(Uₙ)
  fill!(Uₙ₊₁,0.0)
  t = 0
  for i=1:ntime
    Uₙ₊₁ = RK4!(cache, t+Δt, Uₙ, Δt, Kₘₛ, Mₘₛ, fₙ_MS!)  
    Uₙ = Uₙ₊₁
    #    (i%nc == 0) && print("Done t="*string(t+Δt)*"\n")
    t += Δt
  end
  (isnan(sum(Uₙ₊₁))) && print("\nUnstable \n")
  uhsol = zeros(Float64,q*nf+1)
  sol_cache = similar(uhsol)
  cache2 = uhsol, sol_cache
  build_solution!(cache2, Uₙ₊₁, local_basis_vecs)
  uhsol = cache2[1]
  plot!(plt₂, nds_fine, uhsol, label="Approximate sol. (MS Method)", lw=2, lc=:red)
  uexact = [Uₑ(x,tf) for x in nds_fine]  
  plot!(plt₃, nds_fine, uexact, label="Exact solution", lw=2, lc=:green)
end
print("End solving using Multiscale Method.\n\n")

display(plot(plt₁, plt₂, plt₃, layout=(3,1)))

#=
Some benchmarking
=#
#1) RK4 Solution using the direct method
print("Running some benchmarks for 100 iterations...\n")
let
  cache = assembler_cache(nds_fine, elem_fine, quad, q), fn
  Uₙ = U₀.(nds_fine[fn])
  Uₙ₊₁ = similar(Uₙ)
  fill!(Uₙ₊₁,0.0)
  t = 0
  print("Direct Method takes: ")
  @btime begin
    for i=1:100
      Uₙ₊₁ = RK4!($cache, $(t+Δt), $Uₙ, $Δt, $Kϵ[fn,fn], $Mϵ[fn,fn], $fₙ!) 
      Uₙ = Uₙ₊₁
      $(t+=Δt)
    end
  end
  #2) RK4 Solution using the MS method
  Uₙ = setup_initial_condition(U₀, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ)
  Uₙ₊₁ = similar(Uₙ)
  fill!(Uₙ₊₁,0.0)
  t = 0
  cache = contrib_cache, Fₘₛ
  print("Multiscale Method takes: ")
  @btime begin   
    for i=1:100
      Uₙ₊₁ = RK4!($cache, $(t+Δt), $Uₙ, $(Δt), $Kₘₛ, $Mₘₛ, $fₙ_MS!)  
      Uₙ = Uₙ₊₁
      $(t += Δt)
    end
  end
end