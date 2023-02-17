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
A(x) = 0.5
#A(x) = (2.0 + cos((2π*x/(2e-2))))^-1
f(x,t) = 0.0
U₀(x) = sin(π*x)
Uₑ(x,t) = exp(-0.5*π^2*t)*U₀(x)

# Define the necessary parameters
nc = 2^4
nf = 2^11
p = 1
q = 1
l = 5
quad = gausslegendre(4)


# Preallocate all the necessary data
preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
nds_coarse, elems_coarse, nds_fine, elem_fine = fullspace[1:4]
patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch[3:7]
ms_elem = assems[3]
sKms, sFms = multiscale
bc = basis_cache(q)
Uₙ₊₁ = zeros(Float64,q*nf+1)

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
BDFk = 4

print("Begin solving using Direct Method ... \n")
let 
  U = reshape(U₀.(nds_fine[fn]), (q*nf-1,1))
  Uₙ₊ₛ = similar(U)
  fill!(Uₙ₊ₛ,0.0)
  cache = assembler_cache(nds_fine, elem_fine, quad, q), fn
  t = 0
  # Starting steps
  for i=1:BDFk-1
    dlcache = get_dl_cache(i)
    fcache = dlcache, cache
    U₁ = BDFk!(fcache, t+Δt, U, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!, i)     
    U = hcat(U₁, U)
    t += Δt
  end
  # Main BDFk steps
  dlcache = get_dl_cache(BDFk)
  fcache = dlcache, cache
  for i=BDFk:ntime  
    Uₙ₊ₛ = BDFk!(fcache, t+Δt, U, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!, BDFk)
    U[:,2:BDFk] = U[:,1:BDFk-1]
    U[:,1] = Uₙ₊ₛ
    t += Δt
  end
  Uₙ₊₁ = vcat(0.0, Uₙ₊ₛ[:,1], 0.0)
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
  l2err = 0.0
  h1err = 0.0
  Fₘₛ = zeros(Float64,nc*(p+1))
  cache = contrib_cache, Fₘₛ
  U = reshape(setup_initial_condition(U₀, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ), (nc*(p+1),1))
  Uₙ₊ₛ = Vector{Float64}(undef,nc*(p+1))
  fill!(Uₙ₊ₛ,0.0)
  t = 0.0
  # Starting steps
  for i=1:BDFk-1
    dlcache = get_dl_cache(i)
    fcache = dlcache, cache
    U₁ = BDFk!(fcache, t+Δt, U, Δt, Kₘₛ, Mₘₛ, fₙ_MS!, i)
    U = hcat(U₁,U)
    t += Δt
  end
  # Main BDF-k steps
  dlcache = get_dl_cache(BDFk)
  fcache = dlcache, cache
  for i=1:ntime
    copyto!(Uₙ₊ₛ, BDFk!(fcache, t+Δt, U, Δt, Kₘₛ, Mₘₛ, fₙ_MS!, BDFk))
    U[:,2:BDFk] = U[:,1:BDFk-1]
    U[:,1] = Uₙ₊ₛ
    t += Δt
  end
  (isnan(sum(Uₙ₊ₛ))) && print("\nUnstable \n")
  uhsol = zeros(Float64,q*nf+1)
  sol_cache = similar(uhsol)
  cache2 = uhsol, sol_cache
  build_solution!(cache2, Uₙ₊ₛ, local_basis_vecs)
  uhsol = cache2[1]
  plot!(plt₂, nds_fine, uhsol, label="Approximate sol. (MS Method)", lw=2, lc=:red)
  uexact = [Uₑ(x,tf) for x in nds_fine]  
  plot!(plt₃, nds_fine, uexact, label="Exact solution", lw=2, lc=:green)

  # Compute the errors
  bc = basis_cache(q)
  qs,ws=quad    
  for j=1:nf, jj=1:lastindex(qs)
    x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,2]])*0.5 + (0.5*nf^-1)*qs[jj]
    ϕᵢ!(bc,qs[jj])
    l2err += ws[jj]*(dot(Uₙ₊₁[elem_fine[j,:]],bc[3]) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
    ∇ϕᵢ!(bc,qs[jj])
    h1err += ws[jj]*A(x̂)*(dot(Uₙ₊₁[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
  end
  l2err = sqrt(l2err)
  h1err = sqrt(h1err)
  @show l2err, h1err
end
print("End solving using Multiscale Method.\n\n")

display(plot(plt₁, plt₂, plt₃, layout=(3,1)))

#=
Some benchmarking
=#
print("Running some benchmarks for 100 iterations...\n")
function timeit_1()
  dlcache = get_dl_cache(BDFk)
  cache = assembler_cache(nds_fine, elem_fine, quad, q), fn
  U = reshape(U₀.(nds_fine[fn]), (q*nf-1,1))
  t = 0.0
  # Starting steps
  for i=1:BDFk-1
    dlcache = get_dl_cache(i)
    fcache = dlcache, cache
    U₁ = BDFk!(fcache, (t+Δt), U, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!, i)     
    U = hcat(U₁, U)
    t += Δt
  end
  # Main BDFk steps
  dlcache = get_dl_cache(BDFk)
  fcache = dlcache, cache
  for i=BDFk:ntime  
    Uₙ₊ₛ = BDFk!(fcache, t+Δt, U, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!, BDFk)
    U[:,2:BDFk] = U[:,1:BDFk-1]
    U[:,1] = Uₙ₊ₛ
    t += Δt
  end
end
# print("Direct Method takes = ")
# @btime timeit_1(); 
# print("\n")

function timeit_2()
  Fₘₛ = zeros(Float64,nc*(p+1))
  cache = contrib_cache, Fₘₛ
  U = reshape(setup_initial_condition(U₀, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ), (nc*(p+1),1))
  Uₙ₊ₛ = similar(U)
  t = 0
  # Starting steps
  for i=1:BDFk-1
    dlcache = get_dl_cache(i)
    fcache = dlcache, cache
    U₁ = BDFk!(fcache, t+Δt, U, Δt, Kₘₛ, Mₘₛ, fₙ_MS!, i)
    U = hcat(U₁,U)
    t += Δt
  end
  # Main BDFk steps
  dlcache = get_dl_cache(BDFk)
  fcache = dlcache, cache
  for i=1:ntime
    Uₙ₊ₛ = BDFk!(fcache, t+Δt, U, Δt, Kₘₛ, Mₘₛ, fₙ_MS!, BDFk)
    U[:,2:BDFk] = U[:,1:BDFk-1]
    U[:,1] = Uₙ₊ₛ
    t += Δt
  end
end
# print("Multiscale Method takes: ")
# @btime timeit_2(); 
# print("\n")