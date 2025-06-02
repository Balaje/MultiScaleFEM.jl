### Script to implement the schur complement method for solving the coupled system

using LinearAlgebra
import LinearAlgebra: \
    using LinearMaps, IterativeSolvers, LinearAlgebra

struct SchurComplementMatrix{T}
    A11::AbstractMatrix{T}
    A12::AbstractMatrix{T}
    A21::AbstractMatrix{T}
    A22::AbstractMatrix{T}
end
function SchurComplementMatrix(A::AbstractMatrix{T1}, M::T2, N::T2) where {T1,T2<:Integer}
    SchurComplementMatrix(A[1:M, 1:M], A[1:M, M+1:M+N], A[M+1:M+N, 1:M], A[M+1:M+N, M+1:M+N])
end

struct SchurComplementVector{T}
    b1::AbstractVector{T}
    b2::AbstractVector{T}
end
function SchurComplementVector(b::AbstractVector{T1}, M::T2, N::T2) where {T1, T2<:Integer}
    SchurComplementVector(b[1:M], b[M+1:M+N])
end

function \(A::SchurComplementMatrix, b::SchurComplementVector)
    # Matrix Components
    A₁₁ = A.A11
    A₁₂ = A.A12
    A₂₁ = A.A21
    A₂₂ = A.A22
    # Vector Components
    b₁ = b.b1
    b₂ = b.b2
    # Define the solver
    solver = (y,A,b) -> cg!(fill!(y,0.0), A, b; reltol=1e-16, abstol=1e-16)
    # Obtain the Schur Complement system
    A₂₂⁻¹ = InverseMap(A₂₂; solver=solver)
    Σ = A₁₁ - A₁₂*A₂₂⁻¹*A₂₁
    # Solve the Schur Complement system
    Σ⁻¹ = InverseMap(Σ⁻¹; solver=solver)
    F = F₁ - A₁₂*A₂₂⁻¹*F₂    
    U₁ = Σ⁻¹*F₁
    # Obtain the rest of the solution vector
    U₂ = A₂₂⁻¹*(F₂ - A₂₁*U₁)
    [U₁; U₂]
end
