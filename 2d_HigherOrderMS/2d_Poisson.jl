###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl");

domain = (0.0, 1.0, 0.0, 1.0);

# Fine scale space description
nf = 2^8;
q = 1;
nc = 2^3;
p = 1;
l = 3; # Patch size parameter
# A(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1
A(x) = 1.0
f(x) = sin(π*x[1])*sin(π*x[2])

# Fine scale reference solution
model = CartesianDiscreteModel(domain, (nf,nf));
Ω_fine = Triangulation(model);
reffe = ReferenceFE(lagrangian, Float64, 1)
Vh = TestFESpace(Ω_fine, reffe, conformity=:H1, dirichlet_tags="boundary");
Vh0 = TrialFESpace(Vh, 0.0);
dΩ_fine = Measure(Ω_fine, 5);
a(u,v) = ∫(A*(∇(v)⊙∇(u)))dΩ_fine;
b(v) = ∫(v*f)dΩ_fine;
op = AffineFEOperator(a,b,Vh0,Vh);
Uex = solve(op);

# Fine-scale stiffness and mass matrices
V0 = TestFESpace(Ω_fine, reffe, conformity=:H1)
K = assemble_stima(V0, A, 5);
F = assemble_loadvec(V0, f, 5);

# Coarse scale discretization
model_coarse = CartesianDiscreteModel(domain, (nc,nc))
Ω_coarse = Triangulation(model_coarse)
# Obtain the coarse-to-fine map
nsteps =  (Int64(log2(nf/nc)))
coarse_to_fine_map = coarsen(model, nsteps); 

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(domain, nf, nc, l);
# Multiscale Space
Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V0, A);
basis_vec_ms = Vₘₛ.basis_vec_ms

# Multiscale Stiffness and RHS
Kₘₛ = basis_vec_ms'*K*basis_vec_ms;
fₘₛ = basis_vec_ms'*F;
solₘₛ = Kₘₛ\fₘₛ;
Uₘₛ = basis_vec_ms*solₘₛ;
Uₘₛʰ = FEFunction(Vₘₛ.Uh, Uₘₛ);      

Λ = basis_vec_ms[:,2];
Φ = FEFunction(Vₘₛ.Uh, Λ);
writevtk(get_triangulation(Φ), "./2d_HigherOrderMS/basis_ms", cellfields=["u(x)"=>Φ]);
writevtk(model_coarse, "./2d_HigherOrderMS/model");

# Compute the Errors
dΩ = Measure(get_triangulation(Vₘₛ.Uh), 4);
L²Error = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ));
H¹Error = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ));