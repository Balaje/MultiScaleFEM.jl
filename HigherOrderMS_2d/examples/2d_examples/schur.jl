### Script to implement the schur complement method for solving the coupled system

using BlockArrays
using LinearMaps
using IterativeSolvers

function BlockSchur(A::AbstractMatrix{T1}, f::AbstractVector{T2}, N::NTuple{2, T3}) where {T1, T2, T3<:Integer}
    N1 = length(N)
    Aᵦ = BlockArray(A, SVector{N1}(N), SVector{N1}(N))
    fᵦ = BlockVector(f, SVector{N1}(N))
    # Obtain the 2x2 components of the matrix
    A₁₁ = Aᵦ[Block(1,1)];    A₁₂ = Aᵦ[Block(1,2)];
    A₂₁ = Aᵦ[Block(2,1)];    A₂₂ = Aᵦ[Block(2,2)];
    # Obtain the 2x1 component of the vector
    f₁ = fᵦ[Block(1)];
    f₂ = fᵦ[Block(2)];
    # Obtain the inverse of the "good" block
    A₁₁⁻¹ = InverseMap(lu(A₁₁))
    # Calculate the Schur complement system and solve it numerically
    Σ = A₂₂ - A₂₁*A₁₁⁻¹*A₁₂
    F = f₂ - A₂₁*A₁₁⁻¹*f₁
    p = Preconditioners.AMGPreconditioner{SmoothedAggregation}(sparse(Σ))
    U₂ = cg(Σ, F; reltol=1e-16, abstol=1e-16, Pl=p, maxiter=2000)
    # Obtain the second part
    U₁ = A₁₁⁻¹*(f₁ - A₁₂*U₂)
    [U₁; U₂]
end

function BlockSchur(A::AbstractMatrix{T1}, f::AbstractVector{T2}, N::NTuple{3, T3}) where {T1, T2, T3<:Integer}
    N1 = length(N)
    Aᵦ = BlockArray(A, SVector{N1}(N), SVector{N1}(N))
    fᵦ = BlockVector(f, SVector{N1}(N))
    # Obtain the 2x2 components of the matrix
    A₁₁ = Aᵦ[Block(1,1)];    A₁₂ = Aᵦ[Block(1,2)];    A₁₃ = Aᵦ[Block(1,3)];
    A₂₁ = Aᵦ[Block(2,1)];    A₂₂ = Aᵦ[Block(2,2)];    A₂₃ = Aᵦ[Block(2,3)];
    A₃₁ = Aᵦ[Block(3,1)];    A₃₂ = Aᵦ[Block(3,2)];    A₃₃ = Aᵦ[Block(3,3)];
    # Obtain the 2x1 component of the vector
    f₁ = fᵦ[Block(1)];
    f₂ = fᵦ[Block(2)];
    f₃ = fᵦ[Block(3)];
    # Obtain the inverse of the "good" block
    A₁₁⁻¹ = InverseMap(lu(A₁₁))
    # Calculate the Schur complement system and solve it numerically
    B₂₂ = A₂₂ - A₂₁*A₁₁⁻¹*A₁₂;   B₂₃ = A₂₃ - A₂₁*A₁₁⁻¹*A₁₃;
    B₃₂ = A₃₂ - A₃₁*A₁₁⁻¹*A₁₂;   B₃₃ = A₃₃ - A₃₁*A₁₁⁻¹*A₁₃;
    F₂ = f₂ - A₂₁*A₁₁⁻¹*f₁
    F₃ = f₃ - A₃₁*A₁₁⁻¹*f₁

    p = Preconditioners.CholeskyPreconditioner(sparse(B₂₂))
    solver!(y,A,b; Pl=p) = cg!(fill!(y,0.0), A, b;
                               reltol=1e-16, abstol=1e-16, Pl=Pl,
                               maxiter=2000)
    B₂₂⁻¹ = InverseMap(B₂₂; solver=solver!)
    Σ₃ = B₃₃ - B₃₂*B₂₂⁻¹*B₂₃
    b₃ = F₃ - B₃₂*B₂₂⁻¹*F₂

    p = Preconditioners.CholeskyPreconditioner(sparse(Σ₃))
    U₃ = cg(Σ₃, b₃; abstol=1e-16, reltol=1e-16, Pl=p, maxiter=2000)
    U₂ = B₂₂⁻¹*(F₂ - B₂₃*U₃)
    U₁ = A₁₁⁻¹*(f₁ - A₁₂*U₂ - A₁₃*U₃)

    [U₁; U₂; U₃]
end
