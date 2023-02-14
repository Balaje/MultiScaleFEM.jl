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
  M⁺::AbstractMatrix{Float64}, M⁻::AbstractMatrix{Float64}, f!::Function)
  fₙ = (Δt)^2/2*(f!(fcache, tₙ) + 2f!(fcache, tₙ+Δt) + f!(fcache, tₙ+2Δt))
  U = M⁺\(2*M⁻*Uₙ₊₁ - M⁺*Uₙ + fₙ)
  U
end
"""
Function to setup the initial condition by evaluating the L² projection on the MS-space.
"""
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