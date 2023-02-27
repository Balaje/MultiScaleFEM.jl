######### ############ ############ ############ ###########
# Compute the rate of convergence of the multiscale method
# For the heat equation
######### ############ ############ ############ ###########

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
# Problem data
# uₜₜ - (c(x)*uₓ)ₓ = f 
domain = (0.0,1.0)
#c(x) = (2.0 + cos(2π*x/(2e-2)))
c(x) = 1.0
#c(x) = (4.0 + cos(2π*x/(2e-2)))
f(x,t) = 0.0
U₀(x) = 0.0
U₁(x) = π*sin(π*x)
Uₑ(x,t) = sin(π*x)*sin(π*t)

# Define the necessary parameters
nf = 2^15
p = 3
q = 1
quad = gausslegendre(6)

# Temporal parameters
Δt = 1e-4
tf = 1.5
ntime = ceil(Int,tf/Δt)
plt = plot()
plt1 = plot()
fn = 2:q*nf

# Build the matrices for the fine scale problem
nds_fine = LinRange(domain[1], domain[2], nf+1)
elem_fine = [elem_conn(i,j) for i=1:nf, j=0:1]
# Fill the final-scale matrix vector system
assem_cache = assembler_cache(nds_fine, elem_fine, quad, q)
fillsKe!(assem_cache, c)
Kϵ = sparse(assem_cache[5][1], assem_cache[5][2], assem_cache[5][3])
fillsMe!(assem_cache, x->1.0)
Mϵ = sparse(assem_cache[5][1], assem_cache[5][2], assem_cache[5][3])
# The RHS-vector as a function of t
function fₙ!(fcache, tₙ::Float64)  
  # cache, fn = fcache
  # fillsFe!(cache, y->f(y,tₙ))
  # F = collect(sparsevec(cache[6][1], cache[6][2]))
  # F[fn]
  ####   ####   ####   ####   ####   ####  ####
  #### NOTE: This works only if f(x,t) ≡ 0 ####
  ####   ####   ####   ####   ####   ####  ####
  zeros(Float64,size(fn,1)) 
end
Uϵₙ₊₂ = zeros(Float64,q*nf+1)
# Solve the wave equation using the Crank Nicolson Method
let  
  cache = assembler_cache(nds_fine, elem_fine, quad, q), fn
  Uₙ = U₀.(nds_fine[fn])
  Vₙ = U₁.(nds_fine[fn])
  Uₙ₊₁ = CN1!(cache, Uₙ, Vₙ, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!)
  U = similar(Uₙ₊₁)
  fill!(U,0.0)
  t = Δt
  for i=2:ntime
    U = CN!(cache, t, Uₙ, Uₙ₊₁, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!)
    Uₙ = Uₙ₊₁
    Uₙ₊₁ = U
    (i%1000 == 0) && print("Done t="*string(t+Δt)*"\n")
    t+=Δt
  end  
  copyto!(Uϵₙ₊₂, vcat(0.0, U, 0.0))  
end
plt3 = plot(nds_fine, Uϵₙ₊₂, label="Exact solution", lw=2, lc=:black)
# Uₙ₊₂  is the exact solution

𝒩 = [1,2,4,8,16,32,64,128]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

for l in [4,5,6,7,8]
  fill!(L²Error,0.0)
  fill!(H¹Error,0.0)
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    
    let
      preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
      
      fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
      nds_coarse, elems_coarse, nds_fine, elem_fine = fullspace[1:4]
      patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch[3:7]
      ms_elem = assems[3]
      sKms, sFms = multiscale
      bc = basis_cache(q)
      
      cache = bc, zeros(Float64,p+1), quad, preallocated_data
      compute_ms_basis!(cache, nc, q, p, c)
      
      # RHS Function
      function fₙ_MS!(cache, tₙ::Float64)
        contrib_cache, Fms = cache
        # vector_cache = vec_contribs!(contrib_cache, y->f(y,tₙ))
        # fcache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
        # fillsFms!(sFms, fcache, nc, p, l)
        # assemble_MS_vector!(Fms, sFms, ms_elem)
        # Fms        
        ####   ####   ####   ####   ####   ####  ####
        #### NOTE: This works only if f(x,t) ≡ 0 ####
        ####   ####   ####   ####   ####   ####  ####
        0*Fms
      end
      
      # Compute the Stiffness and Mass Matrices
      contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
      matrix_cache = mat_contribs!(contrib_cache, c)
      cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
      fillsKms!(sKms, cache, nc, p, l)
      sMms = similar(sKms)
      for i=1:nc
        sMms[i] = zeros(Float64,size(sKms[i]))    
      end
      matrix_cache = mat_contribs!(contrib_cache, x->1.0; matFunc=fillsMe!)
      cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
      fillsKms!(sMms, cache, nc, p, l)
      Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
      Mₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
      assemble_MS_matrix!(Kₘₛ, sKms, ms_elem)
      assemble_MS_matrix!(Mₘₛ, sMms, ms_elem)
      let
        Fₘₛ = zeros(Float64,nc*(p+1))
        cache = contrib_cache, Fₘₛ
        Uₙ = setup_initial_condition(U₀, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ)
        Vₙ = setup_initial_condition(U₁, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ)
        Uₙ₊₁ = CN1!(cache, Uₙ, Vₙ, Δt, Kₘₛ, Mₘₛ, fₙ_MS!)
        U = similar(Uₙ)
        fill!(U, 0.0)
        t = Δt
        for i=2:ntime
          U = CN!(cache, t, Uₙ, Uₙ₊₁, Δt, Kₘₛ, Mₘₛ, fₙ_MS!)
          Uₙ = Uₙ₊₁
          Uₙ₊₁ = U
          (i%1000 == 0) && print("Done t="*string(t+Δt)*"\n")
          t += Δt
        end        
        uhsol = zeros(Float64,q*nf+1)
        sol_cache = similar(uhsol)
        cache2 = uhsol, sol_cache
        build_solution!(cache2, U, local_basis_vecs)
        uhsol = cache2[1]
        
        bc = basis_cache(q)
        qs,ws=quad    
        for j=1:nf, jj=1:lastindex(qs)
          x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,2]])*0.5 + (0.5*nf^-1)*qs[jj]
          ϕᵢ!(bc,qs[jj])
          L²Error[itr] += ws[jj]*(dot(Uϵₙ₊₂[elem_fine[j,:]],bc[3]) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
          ∇ϕᵢ!(bc,qs[jj])
          H¹Error[itr] += ws[jj]*(dot(Uϵₙ₊₂[elem_fine[j,:]],bc[3])*(2*nf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
        end
        L²Error[itr] = sqrt(L²Error[itr])
        H¹Error[itr] = sqrt(H¹Error[itr])
        
        println("Done nc = "*string(nc))    

        plot!(plt3, nds_fine, uhsol, label="", lw=1, ls=:dash)
      end    
    end
  end
  
  println("Done l = "*string(l))
  plot!(plt, 1 ./𝒩, L²Error, label="(p="*string(p)*"), L² (l="*string(l)*")", lw=2)
  plot!(plt1, 1 ./𝒩, H¹Error, label="(p="*string(p)*"), Energy (l="*string(l)*")", lw=2)
  scatter!(plt, 1 ./𝒩, L²Error, label="", markersize=2)
  scatter!(plt1, 1 ./𝒩, H¹Error, label="", markersize=2, legend=:best)
end
plot!(plt1, 1 ./𝒩, (1 ./𝒩).^(p+2), label="Order "*string(p+2), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)
plot!(plt, 1 ./𝒩, (1 ./𝒩).^(p+3), label="Order "*string(p+3), ls=:dash, lc=:black,  xaxis=:log10, yaxis=:log10)