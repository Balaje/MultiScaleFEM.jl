###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl");

domain = (0.0, 1.0, 0.0, 1.0);

# Fine scale space description
nf = 2^7;
q = 1;
nc = 2^1;
p = 1;
l = 3; # Patch size parameter

Ωms = MultiScaleTriangulation(domain, nf, nc, l);

D = CellField(1.0, Ωms.Ωf);
Ums = MultiScaleFESpace(Ωms, q, p, D, 4);

# Solve the Poisson equation using multiscale method
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2]);
Fϵ = assemble_loadvec(Ums.Uh, f, 4);
Kϵ = assemble_stima(Ums.Uh, D, 4);

# Use the new bases to transform the matrix and vector to the multiscale space.
B = Ums.basis_vec_ms;
Kₘₛ = assemble_ms_matrix(B, p, Kϵ);
Fₘₛ = assemble_ms_load(B, p, Fϵ);
sol = Kₘₛ\Fₘₛ;

# Transform the multiscale solution to the fine-scale
ums = get_fine_scale(B, p, sol);

# # Get the Gridap version for visualization
uH = FEFunction(Ums.Uh, ums);
function visualize_basis_vector(fespace::FESpace, basis_vec_ms, inds)
  for i in inds
    bi = FEFunction(fespace, basis_vec_ms[i][:,1])    
    writevtk(get_triangulation(fespace), "./2d_HigherOrderMS/multiscale-bases-"*string(i), cellfields=["Λᵐˢ"=>bi])
  end
end
# Write paraview functions for visualization
visualize_basis_vector(Ums.Uh, B, [1,4,7])
writevtk(Ωms.Ωf, "./2d_HigherOrderMS/sol_ms", cellfields=["u(x)"=>uH]);