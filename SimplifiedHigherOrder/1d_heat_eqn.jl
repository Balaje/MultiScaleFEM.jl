##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### Julia program to solve a time-dependent problem #####
##### ##### ##### ##### ##### ##### ##### ##### ##### #####
include("basis_functions.jl")
include("assemble_matrices.jl")
include("preallocate_matrices.jl")

#=
Problem data 2: Oscillatory diffusion coefficient
=#
domain = (0.0,1.0)
A(x) = 1.0
f(x,t) = 0.0
U₀(x) = sin(π*x)
Uₑ(x,t) = exp(-A(x)*π^2*t)*U₀(x)

# We solve the time-dependent problem using RK4
function RK4!(fcache, tₙ::Float64, Uₙ::AbstractVector{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function)
  k₁ = (Δt)*(M\(f!(fcache, tₙ) - K*(Uₙ)))
  k₂ = (Δt)*(M\(f!(fcache, tₙ+0.5*Δt) - K*(Uₙ + 0.5*k₁)))
  k₃ = (Δt)*(M\(f!(fcache, tₙ+0.5*Δt) - K*(Uₙ + 0.5*k₂)))
  k₄ = (Δt)*(M\(f!(fcache, tₙ+Δt) - K*(Uₙ + 0.5*k₃)))
  U = Uₙ + (1.0/6.0)*k₁ + (1.0/3.0)*k₂ + (1.0/3.0)*k₂ + (1.0/6.0)*k₄
  U
end 

# Define the necessary parameters
nc = 2^3
nf = 2^6
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
Δt = 1e-5
tf = 1.0
ntime = ceil(Int,tf/Δt)
plt = plot()
let 
  fn = 2:q*nf
  Uₙ = U₀.(nds_fine[fn])
  Uₙ₊₁ = similar(Uₙ)
  fill!(Uₙ₊₁, 0.0)
  t = 0
  cache = assembler_cache(nds_fine, elem_fine, quad, q), fn
  for i=1:ntime
    Uₙ₊₁ = RK4!(cache, t+Δt, Uₙ, Δt, Kϵ[fn,fn], Mϵ[fn,fn], fₙ!)  
    Uₙ = Uₙ₊₁
    (i%1000 == 0) && print("Done t="*string(t+Δt)*"\n")
    t += Δt
  end
  Uₙ₊₁ = vcat(0, Uₙ₊₁, 0)
  plot!(plt, nds_fine, Uₙ₊₁, label="Approximate sol. (direct method)")
  uexact = [Uₑ(x,tf) for x in nds_fine]  
  plot!(plt, nds_fine, uexact, label="Exact solution")
end


# Compute the Multiscale basis
cache = bc, zeros(Float64,p+1), quad, preallocated_data
compute_ms_basis!(cache, nc, q, p, A)

function fₙ_MS!(cache, tₙ::Float64)
  contrib_cache, Fms = cache
  vector_cache = vec_contribs!(contrib_cache, y->f(y,tₙ))
  fcache = local_basis_vecs, elem_indices_to_global_indices, Lᵀ, vector_cache
  fillsFms!(sFms, fcache, nc, p, l)
  assemble_MS_vector!(Fms, sFms, ms_elem)
  Fms
end

let 
  contrib_cache = mat_vec_contribs_cache(nds_fine, elem_fine, q, quad, elem_indices_to_global_indices)
  matrix_cache = mat_contribs!(contrib_cache, A)
  cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
  fillsKms!(sKms, cache, nc, p, l)
  ## = The mass matrix
  sMms = similar(sKms)
  for i=1:nc
    sMms[i] = zeros(Float64,size(sKms[i]))    
  end
  matrix_cache = mat_contribs!(contrib_cache, A; matFunc=fillsMe!)
  cache = local_basis_vecs, elem_indices_to_global_indices, L, Lᵀ, matrix_cache, ipcache
  fillsKms!(sMms, cache, nc, p, l)
  ## =
  Kₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
  global Mₘₛ = zeros(Float64,nc*(p+1),nc*(p+1))
  Fₘₛ = zeros(Float64,nc*(p+1))
  assemble_MS_matrix!(Kₘₛ, sKms, ms_elem)
  assemble_MS_matrix!(Mₘₛ, sMms, ms_elem)
  cache = contrib_cache, Fₘₛ
  global Uₙ = setup_initial_condition(U₀, nds_fine, nc, nf, local_basis_vecs, quad, p, q, Mₘₛ)
  Uₙ₊₁ = similar(Uₙ)
  fill!(Uₙ₊₁,0.0)
  t = 0
  for i=1:ntime
    Uₙ₊₁ = RK4!(cache, t+Δt, Uₙ, Δt, Kₘₛ, Mₘₛ, fₙ_MS!)  
    Uₙ = Uₙ₊₁
    (i%1000 == 0) && print("Done t="*string(t+Δt)*"\n")
    t += Δt
  end
  uhsol = zeros(Float64,q*nf+1)
  sol_cache = similar(uhsol)
  cache2 = uhsol, sol_cache
  build_solution!(cache2, Uₙ₊₁, local_basis_vecs)
  uhsol = cache2[1]
  plot!(plt, nds_fine, uhsol, label="Approximate sol. (MS Method)")
end


function setup_initial_condition(U₀::Function, nds::AbstractVector{Float64}, nc::Int64, nf::Int64, 
  local_basis_vecs::Vector{Matrix{Float64}}, quad::Tuple{Vector{Float64},Vector{Float64}}, 
  p::Int64, q::Int64, massmat::Matrix{Float64})
  qs,ws = quad
  U0 = Matrix{Float64}(undef,p+1,nc)
  bc = basis_cache(q)
  for t=1:nc
    lb = local_basis_vecs[t]
    for qi=1:lastindex(qs)    
      ϕᵢ!(bc,qs[qi])                
      for i=1:q*nf, j=1:p+1, k=1:q+1             
        x̂ = (nds[i+1]+nds[i])*0.5 + (nds[i+1]-nds[i])*0.5*qs[qi]
        U0[j,t] += ws[qi] * (bc[3][k] * lb[i,j]) * U₀(x̂) * (nds[i+1]-nds[i])*0.5
      end
    end
  end
  massmat\vec(U0)
end