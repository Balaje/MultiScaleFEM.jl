using Plots

include("meshes.jl");
include("assemblers.jl");
include("fespaces.jl");
include("basis_functions.jl")
include("local_matrix_vector.jl")
include("assemble_matrices.jl")

#A(x) = @. 1
ε = 2^-7
A(x) = @. (2 + cos(2π*x/ε))^-1

p = 1
q = 1
l = 3
n = 20
nₚ = 200
Ω = 𝒯((0,1),n)

# - Pre-Compute the assembler using the first patch size.
# - The patch size remains constant in the local problems
patch = Ω[1:2l+1]
patch_mesh = 𝒯((patch.nds[1], patch.nds[end]), nₚ)
Kₐ = MatrixAssembler(H¹ConformingSpace(), q, patch_mesh.elems)
Lₐ = MatrixAssembler(H¹ConformingSpace(), L²ConformingSpace(), (q,p), (patch_mesh.elems, patch.elems))
Fₐ = VectorAssembler(L²ConformingSpace(), p, patch.elems)
Rₛ = compute_basis_functions(Ω, A, (q,p), [Kₐ,Lₐ], [Fₐ]; qorder=10, Nfine=nₚ)
## Plot to verify the basis functions
plt = plot()
el = 10;
for local_basis = 1:p+1
  plot!(plt, Rₛ[el][local_basis].nds, Rₛ[el][local_basis].Λ,
        label="Element el="*string(el)*" Local Basis i="*string(local_basis), lw=2)
end
xlims!(plt, (0,1))
