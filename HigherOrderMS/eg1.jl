using Plots

include("meshes.jl");
include("fespaces.jl");
include("assemblers.jl");
include("local_matrix_vector.jl")
include("assemble_matrices.jl")
include("solve_local_problems.jl");

#A(x) = @. 1
ε = 2^-6
A(x) = @. (2 + cos(2π*x/ε))^-1

p = 1
q = 1
l = 1
n = 5
nₚ = 100
Ω = 𝒯((0,1),n) 

# - Pre-Compute the assembler using the first patch size. 
# - The patch size remains constant in the local problems
patch = Ω[1:2l+1]
patch_mesh = 𝒯((patch.nds[1], patch.nds[end]), nₚ)
Kₐ = MatrixAssembler(H¹ConformingSpace(), q, patch_mesh.elems)
Lₐ = MatrixAssembler(H¹ConformingSpace(), L²ConformingSpace(), (q,p), (patch_mesh.elems, patch.elems))
Fₐ = VectorAssembler(L²ConformingSpace(), p, patch.elems)

# Now to compute the new basis functions on a patch
function compute_basis_functions!(
    Rₛ::Vector{Vector{Rˡₕ}}, 
    Ω::T, A::Function, fespace, MatAssems::Vector{MatrixAssembler}, 
    VecAssems::Vector{VectorAssembler};
    qorder=3, Nfine=nₚ) where T<:MeshType    

    q,p = fespace
    n = size(Ω.elems,1)
    Kₐ, Lₐ = MatAssems
    Fₐ, = VecAssems
    for el=1:n
        # Get the start and last index of the patch
        start = (el-l)<1 ? 1 : el-l; last = start+2l
        last = (last>n) ? n : last; start = last-2l
        NˡK = Ω[start:last]
        Ωₚ = (NˡK.nds[1], NˡK.nds[end])
        elem = Ω.nds[Ω.elems[el,1]:Ω.elems[el,2]]
        NˡKₕ = 𝒯(Ωₚ, Nfine)
        VₕᵖNˡK = L²Conforming(NˡK, p); # Coarse Mesh
        H¹₀NˡK = H¹Conforming(NˡKₕ ,q, [1,nₚ+1]); # Fine Mesh
        function Bₖ(x,nds)
            a,b=nds
            x̂ = -(a+b)/(b-a) + 2/(b-a)*x
            (a ≤ x ≤ b) ? VₕᵖNˡK.basis(x̂) : zeros(Float64,VₕᵖNˡK.p+1)
        end
        R = map(i->Rˡₕ(x->Bₖ(x,elem)[i], A, (H¹₀NˡK, VₕᵖNˡK), [Kₐ,Lₐ], [Fₐ]; qorder=qorder),
                1:p+1)
        Rₛ[el] = R
    end
end

Rₛ = Vector{Vector{Rˡₕ}}(undef,n)
compute_basis_functions!(Rₛ, Ω, A, (q,p), [Kₐ,Lₐ], [Fₐ]; qorder=10, Nfine=nₚ)
## Plot to verify the basis functions 
plt = plot()
el = 3; 
for local_basis = 1:p+1
    plot!(plt, Rₛ[el][local_basis].nds, Rₛ[el][local_basis].Λ, 
    label="Element el="*string(el)*" Local Basis i="*string(local_basis), lw=2)    
end
xlims!(plt, (0,1))