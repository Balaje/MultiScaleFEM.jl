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
domain = (0.0,1.0)
A(x) = 0.5
f(x,t) = 0.0
U₀(x) = sin(π*x)
∇U₀(x) = π*cos(π*x)
Uₑ(x,t) = exp(-0.5*π^2*t)*U₀(x)
∇Uₑ(x,t) = exp(-0.5*π^2*t)*∇U₀(x)

# Define the necessary parameters
nf = 2^11
p = 1
q = 1
quad = gausslegendre(4)

# Temporal parameters
Δt = 1e-4
tf = 100*Δt
ntime = ceil(Int,tf/Δt)
plt = plot()
plt1 = plot()
fn = 2:q*nf

𝒩 = [1,2,4,8,16]
L²Error = zeros(Float64,size(𝒩))
H¹Error = zeros(Float64,size(𝒩))

for l in [2,3,4,5,6]
  for (nc,itr) in zip(𝒩,1:lastindex(𝒩))
    
    let
      preallocated_data = preallocate_matrices(domain, nc, nf, l, (q,p))
      
      fullspace, fine, patch, local_basis_vecs, mats, assems, multiscale = preallocated_data
      nds_coarse, elems_coarse, nds_fine, elem_fine, assem_H¹H¹ = fullspace
      nds_fineₛ, elem_fineₛ = fine
      nds_patchₛ, elem_patchₛ, patch_indices_to_global_indices, elem_indices_to_global_indices, L, Lᵀ, ipcache = patch
      sKeₛ, sLeₛ, sFeₛ, sLVeₛ = mats
      assem_H¹H¹ₛ, assem_H¹L²ₛ, ms_elem = assems
      sKms, sFms = multiscale
      bc = basis_cache(q)
      
      cache = bc, zeros(Float64,p+1), quad, preallocated_data
      compute_ms_basis!(cache, nc, q, p, A)
      
      # RHS Function
      function fₙ_MS!(cache, tₙ::Float64)
        contrib_cache, Fms = cache
        vector_cache = vec_contribs!(contrib_cache, y->f(y,tₙ))
        fcache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
        fillsFms!(sFms, fcache, nc, p, l)
        assemble_MS_vector!(Fms, sFms, ms_elem)
        Fms
      end
      
      # Compute the Stiffness and Mass Matrices
      contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
      matrix_cache = mat_contribs!(contrib_cache, A)
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
        Uₙ₊₁ = similar(Uₙ)
        fill!(Uₙ₊₁,0.0)
        t = 0.0
        for i=1:ntime
          Uₙ₊₁ = RK4!(cache, t, Uₙ, Δt, Kₘₛ, Mₘₛ, fₙ_MS!)  
          Uₙ = Uₙ₊₁
          (i%1000 == 0) && print("Done t="*string(t+Δt)*"\n")
          t += Δt
        end
        (isnan(sum(Uₙ₊₁))) && exit(1)
        uhsol = zeros(Float64,q*nf+1)
        sol_cache = similar(uhsol)
        cache2 = uhsol, sol_cache
        build_solution!(cache2, Uₙ₊₁, local_basis_vecs)
        uhsol = cache2[1]

        bc = basis_cache(q)
        qs,ws=quad    
        for j=1:nf, jj=1:lastindex(qs)
          x̂ = (nds_fine[elem_fine[j,1]] + nds_fine[elem_fine[j,2]])*0.5 + (0.5*nf^-1)*qs[jj]
          ϕᵢ!(bc,qs[jj])
          L²Error[itr] += ws[jj]*(Uₑ(x̂, tf) - dot(uhsol[elem_fine[j,:]],bc[3]))^2*(0.5*nf^-1)
          ∇ϕᵢ!(bc,qs[jj])
          H¹Error[itr] += ws[jj]*A(x̂)*(∇Uₑ(x̂, tf) - dot(uhsol[elem_fine[j,:]],bc[3])*(2*nf))^2*(0.5*nf^-1)
        end
        L²Error[itr] = sqrt(L²Error[itr])
        H¹Error[itr] = sqrt(H¹Error[itr])
          
        println("Done nc = "*string(nc))    
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