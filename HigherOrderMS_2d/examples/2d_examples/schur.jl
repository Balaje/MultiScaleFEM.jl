### Script to implement the schur complement method for solving the coupled system

using LinearAlgebra
import LinearAlgebra: \
    using LinearMaps, IterativeSolvers, LinearAlgebra
import Base: +, *, collect

struct SchurComplementMatrix{T} <: AbstractMatrix{T}
    A11::AbstractMatrix{T}
    A12::AbstractMatrix{T}
    A21::AbstractMatrix{T}
    A22::AbstractMatrix{T}
end
function SchurComplementMatrix(A::AbstractMatrix{T1}, M::T2, N::T2) where {T1,T2<:Integer}
    SchurComplementMatrix(A[1:M, 1:M], A[1:M, M+1:M+N], A[M+1:M+N, 1:M], A[M+1:M+N, M+1:M+N])
end


struct SchurComplementVector{T} <: AbstractMatrix{T}
    b1::AbstractVector{T}
    b2::AbstractVector{T}
end
function SchurComplementVector(b::AbstractVector{T1}, M::T2, N::T2) where {T1, T2<:Integer}
    SchurComplementVector(b[1:M], b[M+1:M+N])
end

### ###### ###### ###### ###### ###### ###### ###### ###
# Export some methods for matrix vector operators
### ###### ###### ###### ###### ###### ###### ###### ###
"""
Sum of two Schur Complement Matrices
"""
function +(A::SchurComplementMatrix, B::SchurComplementMatrix)
    SchurComplementMatrix((A.A11+B.A11), (A.A12+B.B12), (A.A21+B.A21), (A.A22+B.B22))
end

"""
Matrix-vector product of a Schur Complement system
"""
function *(A::SchurComplementMatrix, b::SchurComplementVector)
    SchurComplementVector((A.A11*b.b1+A.A12*b.b2), (A.A21*b.b1+A.A22*b.b22))
end

"""
Scalar Multiplication of a Schur Complement Matrix
"""
function *(c::T, A::SchurComplementMatrix) where T<:Number
    SchurComplementMatrix(c*A.A11, c*A.A12, c*A.A21, c*A.A22)
end

"""
Scalar Multiplication of a Schur Complement vector
"""
function *(c::T, b::SchurComplementVector) where T<:Number
    SchurComplementVector(c*b.b1, b*b.b2)
end

"""
Convert Schur Complement Vector to a raw vector
"""
function collect(x::SchurComplementVector)
    [x.b1; x.b2]
end

"""
Convert Schur Complement Matrix to a raw matrix
"""
function collect(x::SchurComplementMatrix)
    [x.A11 x.A12;
     x.A21 x.A22]
end

"""
Product of a Schur-Complement Matrix and a raw vector
"""
function *(A::SchurComplementMatrix, b::Vector{T}) where T
    collect(A)*b
end

"""
Solve a Schur Complement system
"""
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

"""
Solve a Schur Complement system with a schur complement matrix and a raw vector
"""
function \(A::SchurComplementMatrix, b::Vector{T}) where T
    M = size(A.A11,1)
    N = size(A.A22,1)
    A\(SchurComplementVector(b, M, N))
end



