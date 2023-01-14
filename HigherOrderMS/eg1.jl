using Plots

include("meshes.jl");
include("fespaces.jl");
include("assemblers.jl");
include("local_matrix_vector.jl")
include("assemble_matrices.jl")
include("solve_local_problems.jl");

#A(x) = @. 1
ε = 1e-2
A(x) = @. (2 + cos(2π*x/ε))^-1
N₁ = 5; N₂ = 200;
mesh₁ = 𝒯((0.0,0.8),4) # nel = l+1
mesh₂ = 𝒯((0.0,0.8),N₂)
U = L²Conforming(mesh₁,1); # Coarse
V = H¹Conforming(mesh₂,1,[1,N₂+1]); # Fine
assem₁ = MatrixVectorAssembler(V,V)
assem₂ = MatrixVectorAssembler(V,U)
function Bₖ(x,nds)
  a,b=nds
  x̂ = -(a+b)/(b-a) + 2/(b-a)*x
  (a ≤ x ≤ b) ? U.Λₖᵖ(x̂) : zeros(Float64,U.p+1)
end
R = Rˡₕ(y->Bₖ(y,(0.0,0.2))[2], A, assem₁, assem₂; qorder=3)
plt = plot(R.nds, R.Λ⃗)
xlims!(plt, (0,1))
