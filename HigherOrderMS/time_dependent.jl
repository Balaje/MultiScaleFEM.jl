##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##
##### Script that contains the routines to implement the time dependent problems #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##
"""
The RK-4 implementation for solving the transient linear heat equation
"""
function RK4!(fcache, tₙ::Float64, Uₙ::AbstractVector{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function)
  k₁ = (Δt)*(M\(f!(fcache, tₙ) - K*(Uₙ)))
  k₂ = (Δt)*(M\(f!(fcache, tₙ+0.5*Δt) - K*(Uₙ + 0.5*k₁)))
  k₃ = (Δt)*(M\(f!(fcache, tₙ+0.5*Δt) - K*(Uₙ + 0.5*k₂)))
  k₄ = (Δt)*(M\(f!(fcache, tₙ+Δt) - K*(Uₙ + 0.5*k₃)))
  U = Uₙ + (1.0/6.0)*k₁ + (1.0/3.0)*k₂ + (1.0/3.0)*k₂ + (1.0/6.0)*k₄
  U
end
"""
The Crank-Nicolson scheme for solving the transient linear wave equation
"""
function CN!(fcache, tₙ::Float64, Uₙ::AbstractVector{Float64}, Uₙ₊₁::AbstractVector{Float64}, Δt::Float64,
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function)
  NM!(fcache, tₙ, Uₙ, Uₙ₊₁, Δt, K, M, f!, 1/4, 1/2)
end
function CN1!(fcache, U₀::AbstractVector{Float64}, V₀::AbstractVector{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function)
  NM1!(fcache, U₀, V₀, Δt, K, M, f!, 1/4, 1/2)
end
"""
The Leap-Frog scheme for solve the transient linear wave equation
"""
function LF!(fcache, tₙ::Float64, Uₙ::AbstractVector{Float64}, Uₙ₊₁::AbstractVector{Float64}, Δt::Float64,
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function)
  NM!(fcache, tₙ, Uₙ, Uₙ₊₁, Δt, K, M, f!, 0.0, 1/2)
end
function LF1!(fcache, U₀::AbstractVector{Float64}, V₀::AbstractVector{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function)
  NM1!(fcache, U₀, V₀, Δt, K, M, f!, 0.0, 1/2)
end
"""
General Newmark scheme for the wave equation
"""
function NM!(fcache, tₙ::Float64, Uₙ::AbstractVector{Float64}, Uₙ₊₁::AbstractVector{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function, β::Float64, γ::Float64)
  M⁺ = (M + Δt^2*(β)*K)
  M⁻ = (M - Δt^2/4*(1-4β+2γ)*K)
  M̃  = (M + Δt^2/2*(1+2β-2γ)*K)
  Fₙ = Δt^2*(f!(fcache,tₙ-Δt)+f!(fcache,tₙ))*0.5
  M⁺\(2*M⁻*Uₙ₊₁ - M̃ *Uₙ + Fₙ)
end
function NM1!(fcache, U₀::AbstractVector{Float64}, V₀::AbstractVector{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function, β::Float64, γ::Float64)
  M⁺ = (2M + Δt^2/2*(1+4β-2γ)*K)
  M⁻ = (M - Δt^2/4*(1-4β+2γ)*K)
  M̃  = (M + Δt^2/2*(1+2β-2γ)*K)
  Fₙ = Δt^2*(f!(fcache,0.0)+f!(fcache,Δt))*0.5
  M⁺\(2*M⁻*U₀ + 2*Δt*M̃ *V₀ + Fₙ)
end
"""
The Backward Difference Formula of order k for the linear heat equation
"""
function BDFk!(cache, tₙ::Float64, U::AbstractVecOrMat{Float64}, Δt::Float64, 
  K::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, f!::Function, k::Int64)
  # U should be arranged in descending order (n+k), (n+k-1), ...
  @assert (size(U,2) == k) # Check if it is the right BDF-k
  dl_cache, fcache = cache
  coeffs = dl!(dl_cache, k)
  RHS = 1/coeffs[k+1]*(Δt)*(f!(fcache, tₙ+k*Δt))  
  for i=0:k-1
    RHS += -(coeffs[k-i]/coeffs[k+1])*M*U[:,i+1]
  end 
  LHS = (M + 1.0/(coeffs[k+1])*Δt*K)
  Uₙ₊ₖ = LHS\RHS
  Uₙ₊ₖ
end
function get_dl_cache(k::Int64)
  0, 0, zeros(Float64,k+1)
end
function dl!(cache, k::Int64)
  sum, prod, res = cache
  fill!(res,0.0)
  xⱼ = 0.0:Float64(k)
  for i=1:k+1
    res[i] = dlₖ!((sum,prod), Float64(k), xⱼ, i)
  end
  res
end 
function dlₖ!(cache, t::Float64, tⱼ::AbstractVector{Float64}, j::Int64)
  sum, prod = cache
  sum = 0.0
  prod = 1.0
  for l=1:lastindex(tⱼ)
    (l ≠ j) && begin
      prod = 1/(tⱼ[j]- tⱼ[l])
      for m=1:lastindex(tⱼ)
        (m ≠ j) && (m ≠ l) && begin
          prod = prod*(t - tⱼ[m])/(tⱼ[j]-tⱼ[m])        
        end
      end
      sum += prod
    end     
  end
  sum
end
"""
Function to setup the initial condition by evaluating the L² projection on the MS-space.
"""
function setup_initial_condition(U₀::Function, nds::AbstractVector{Float64}, nc::Int64, nf::Int64, 
  local_basis_vecs::Vector{Matrix{Float64}}, quad::Tuple{Vector{Float64},Vector{Float64}}, 
  p::Int64, q::Int64, massmat::Matrix{Float64})
  qs,ws = quad
  U0 = Matrix{Float64}(undef,p+1,nc)
  fill!(U0,0.0)
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