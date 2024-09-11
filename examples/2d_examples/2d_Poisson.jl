###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 

using Gridap
using MultiscaleFEM
# include("2d_HigherOrderMS.jl");

domain = (0.0, 1.0, 0.0, 1.0);

# Fine scale space description
nf = 2^7;
nc = 2^1;
p = 1;
l = 3; # Patch size parameter
# A(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1
A(x) = 1.0
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2])

# Fine scale reference solution
model = CartesianDiscreteModel(domain, (nf,nf));
Ω_fine = Triangulation(model);
reffe = ReferenceFE(lagrangian, Float64, 1)
Uex = CellField(x->sin(π*x[1])*sin(π*x[2]), Ω_fine)

# Coarse scale discretization
model_coarse = CartesianDiscreteModel(domain, (nc,nc))
Ω_coarse = Triangulation(model_coarse)
# Obtain the coarse-to-fine map
nsteps =  (Int64(log2(nf/nc)))
coarse_to_fine_map = coarsen(model, nsteps); 

# # Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(domain, nf, nc, l);
# # Multiscale Space
Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, TestFESpace(Ω_fine, reffe, conformity=:H1), A, f);
basis_vec_ms = Vₘₛ.basis_vec_ms
K, L, Λ, F = Vₘₛ.fine_scale_system

# # Multiscale Stiffness and RHS
Kₘₛ = basis_vec_ms'*K*basis_vec_ms;
fₘₛ = basis_vec_ms'*F;
solₘₛ = Kₘₛ\fₘₛ;
Uₘₛ = basis_vec_ms*solₘₛ;
Uₘₛʰ = FEFunction(Vₘₛ.Uh, Uₘₛ);      

Λ = basis_vec_ms[:,2];
Φ = FEFunction(Vₘₛ.Uh, Λ);
writevtk(get_triangulation(Φ), "basis_ms", cellfields=["u(x)"=>Φ]);
writevtk(model_coarse, "model");

# Compute the Errors
dΩ = Measure(get_triangulation(Vₘₛ.Uh), 4);
L²Error = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ));
H¹Error = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ));