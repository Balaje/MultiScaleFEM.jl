### Script to implement the schur complement method for solving the coupled system
using LinearAlgebra

struct SchurComplementMatrix{T₁,T₂} <: AbstractMatrix{T₁}
  A::NTuple{4,AbstractMatrix{T₁}}  
  M::NTuple{2,T₂}  
end

function SchurComplementMatrix(A::AbstractMatrix{Float64}, M::NTuple{2,Int64})
  m, n = M
  A₁₁ = A[1:m, 1:m];  A₁ᵧ = A[1:m, 1+m:n+m]
  Aᵧ₁ = A[1+m:n+m, 1:m];  Aᵧᵧ = A[1+m:n+m, 1+m:n+m]  
  SchurComplementMatrix((A₁₁, A₁ᵧ, Aᵧ₁, Aᵧᵧ), M)
end

import Base.\
function \(A::SchurComplementMatrix, f::AbstractVector{Float64})
  m, n = A.M
  f₁ = f[1:m];  fᵧ = f[m+1:m+n]
  A₁₁, A₁ᵧ, Aᵧ₁, Aᵧᵧ = A.A
  Lᵧ₁ = collect(Aᵧ₁)
  L₁ᵧ = collect(A₁ᵧ)
  Lᵧᵧ = LinearAlgebra.lu(Aᵧᵧ)
  Σ = (A₁₁ - L₁ᵧ*(Lᵧᵧ\Lᵧ₁))
  T = f₁ - L₁ᵧ*(Lᵧᵧ\fᵧ)
  U₁ = Σ\T
  Uᵧ = Lᵧᵧ\(fᵧ - Lᵧ₁*U₁)
  vcat(U₁, Uᵧ)
end

import LinearAlgebra.cond
function cond(A::SchurComplementMatrix)
  A₁₁, A₁ᵧ, Aᵧ₁, Aᵧᵧ = A.A
  Lᵧ₁ = collect(Aᵧ₁)
  Lᵧᵧ = LinearAlgebra.lu(Aᵧᵧ)
  Σ = (A₁₁ - A₁ᵧ*(Lᵧᵧ\Lᵧ₁))
  LinearAlgebra.cond(collect(Σ)), LinearAlgebra.cond(collect(Aᵧᵧ))
end

import Base.*
function *(α::T, A::SchurComplementMatrix) where T<:Number  
  SchurComplementMatrix(α.*A.A, A.M)
end
function *(A::SchurComplementMatrix, v::AbstractVector{T}) where T<:Number
  A_mat = [A.A[1] A.A[2]; A.A[3] A.A[4]]
  A_mat*v
end

import Base.+
function +(A::SchurComplementMatrix, B::SchurComplementMatrix)
  @assert A.M == B.M "Matrices must be of the same dimensions"
  SchurComplementMatrix(A.A .+ B.A, A.M)
end

import Base.size
function size(A::SchurComplementMatrix)
  ((A.M[1] + A.M[2]), (A.M[1] + A.M[2]))
end