###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 

# # Run this the first time
# using Pkg
# Pkg.activate(".")

using DoubleFloats
T₁ = Double64

using Gridap
using MultiscaleFEM
using SparseArrays
using ProgressMeter

domain = T₁.((0.0, 1.0, 0.0, 1.0));

(length(ARGS)==4) && begin (nf, nc, p, l) = parse.(Int64, ARGS) end
if(length(ARGS)==0)
  nf = 2^4;
  nc = 2^1;
  p = 1;
  l = 1; # Patch size parameter
end

# Fine Scale discretization
FineScale = FineTriangulation(domain, nf);

# Random field
epsilon = 2^2
repeat_dims = (Int64(nf/epsilon), Int64(nf/epsilon))
a₁,b₁ = T₁.((0.5,1.5))
rand_vals = ones(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)
# Diffusion Coefficient
A = CellField(vec(vals_epsilon), FineScale.trian)

f(x) = 2π^2*sin(π*x[1])*sin(π*x[2]);

reffe = ReferenceFE(lagrangian, T₁, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1; vector_type=Vector{T₁});

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation (Contains both Coarse and Fine scales)
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);

# Assemble the fine-scale matrices
K = assemble_stima(V₀, A, 4; T=T₁);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);
F = assemble_loadvec(V₀, f, 4; T=T₁);

# # Multiscale space without stabilization
# γ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));

# Multiscale space along with stabilization
Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
# Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
γ = StabilizedMultiScaleFESpace(Vₘₛ, p, V₀, (K, L, Λ), domain, A);

# Convert the cell-wise basis function to a sparse matrix
B = zero(L)
build_basis_functions!((B,), (γ,));

# Assemble the multiscale system and solve the problem
Kₘₛ = assemble_ms_matrix(B, K);
Fₘₛ = assemble_ms_loadvec(B, F);
solₘₛ = Kₘₛ\Fₘₛ;
Uₘₛ = B*solₘₛ;
Uₘₛʰ = FEFunction(γ.Uh, Uₘₛ);      
  
# Reference solution
Vh = TestFESpace(Ωₘₛ.Ωf.trian, reffe, conformity=:H1, dirichlet_tags="boundary");
Vh0 = TrialFESpace(Vh, 0.0);
dΩ = Measure(Ωₘₛ.Ωf.trian, 5);
a(u,v) = ∫(D*(∇(v)⊙∇(u)))dΩ;
b(v) = ∫(v*f)dΩ;
op = AffineFEOperator(a,b,Vh0,Vh);
Uex = solve(op);

# Compute the L²-and-H¹- Errors
dΩ = Measure(get_triangulation(γ.Uh), 4);
L²Error = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ))
H¹Error = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ))
println("Done computing the solutions...")
println("L²Error    H¹Error:")
println("$L²Error   $H¹Error;")