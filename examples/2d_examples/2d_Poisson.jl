###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 

using Gridap
using MultiscaleFEM
using ProgressBars
using SparseArrays

# include("2d_HigherOrderMS.jl");

domain = (0.0, 1.0, 0.0, 1.0);

# Fine scale space description
nf = 2^7;
nc = 2^4;
p = 1;
l = 3; # Patch size parameter
# A(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1
A(x) = 1.0
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2])

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, Float64, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1);
K = assemble_stima(V₀, A, 0);
F = assemble_loadvec(V₀, f, 3);

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);
Λ = assemble_rhs_matrix(CoarseScale.trian, p);

# # Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);
C₂F = get_coarse_scale_elem_fine_scale_node_indices(Ωₘₛ);
L = assemble_rect_matrix(CoarseScale.trian, V₀, C₂F, p);

Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
basis_vec_ms = Vₘₛ.basis_vec_ms;
Ks, Ls, Λs = Vₘₛ.fine_scale_system;

# Lazy fill the fine-scale vector
Fs = lazy_fill(F, num_cells(CoarseScale.trian));

elem_to_monomials(i) = (i-1)*(p+1)^2+1:i*(p+1)^2;
L1 = zero(L);
for i = 1:num_cells(CoarseScale.trian)
  L1[:,elem_to_monomials(i)] .= basis_vec_ms[i];
end

# Multiscale Stiffness and RHS
Kₘₛ = assemble_ms_matrix(L1, K);
Fₘₛ = assemble_ms_loadvec(L1, F);
solₘₛ = solve_ms_problem(Kₘₛ, Fₘₛ, ((p+1)^2, num_cells(CoarseScale.trian)));
elem_fine_scale = lazy_map(*, basis_vec_ms,  solₘₛ);
Uₘₛ = assemble_fine_scale_from_ms(elem_fine_scale);
Uₘₛʰ = FEFunction(Vₘₛ.Uh, Uₘₛ);      

# # Compute the Errors
Uex = CellField(x->sin(π*x[1])*sin(π*x[2]), FineScale.trian);
dΩ = Measure(get_triangulation(Vₘₛ.Uh), 4);
L²Error = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ))
H¹Error = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ))

# Λ = basis_vec_ms[2][:,1];
# Φ = FEFunction(Vₘₛ.Uh, Λ);
# writevtk(get_triangulation(Φ), "basis_ms", cellfields=["u(x)"=>Φ]);
# writevtk(CoarseScale.trian, "model");