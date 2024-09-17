###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 

using Pkg
Pkg.activate(".")

using Gridap
using MultiscaleFEM
using SparseArrays
using ProgressMeter

using MPI
comm = MPI.COMM_WORLD
MPI.Init()
mpi_size = MPI.Comm_size(comm)
mpi_rank = MPI.Comm_rank(comm)

domain = (0.0, 1.0, 0.0, 1.0);

# Fine scale space description
(length(ARGS)==4) && begin (nf, nc, p, l) = parse.(Int64, ARGS) end
if(length(ARGS)==0)
  nf = 2^7;
  nc = 2^4;
  p = 1;
  l = 5; # Patch size parameter
end
# A(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1
A(x) = 1.0
f(x) = sin(3π*x[1])*sin(5π*x[2])

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, Float64, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1);
K = assemble_stima(V₀, A, 4);
F = assemble_loadvec(V₀, f, 4);

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
basis_vec_ms = Vₘₛ.basis_vec_ms;
Ks, Ls, Λs = Vₘₛ.fine_scale_system;

(mpi_rank == 0) && println("Computing basis functions...")
t1 = MPI.Wtime()
B = zero(L)
build_basis_functions!((B,), (Vₘₛ,), comm);
t2 = MPI.Wtime()
(mpi_rank == 0) && println("Elasped time = $(t2-t1)\n");

if(mpi_rank == 0)
  Kₘₛ = assemble_ms_matrix(B, K);
  Fₘₛ = assemble_ms_loadvec(B, F);
  solₘₛ = Kₘₛ\Fₘₛ;
  Uₘₛ = B*solₘₛ;
  Uₘₛʰ = FEFunction(Vₘₛ.Uh, Uₘₛ);      
  
  Vh = TestFESpace(Ωₘₛ.Ωf.trian, reffe, conformity=:H1, dirichlet_tags="boundary");
  Vh0 = TrialFESpace(Vh, 0.0);
  dΩ = Measure(Ωₘₛ.Ωf.trian, 5);
  a(u,v) = ∫(A*(∇(v)⊙∇(u)))dΩ;
  b(v) = ∫(v*f)dΩ;
  op = AffineFEOperator(a,b,Vh0,Vh);
  Uex = solve(op);

  # Uex = CellField(x->sin(π*x[1])*sin(π*x[2]), FineScale.trian);
  dΩ = Measure(get_triangulation(Vₘₛ.Uh), 4);
  L²Error = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ))
  H¹Error = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ))
  println("$L²Error, $H¹Error")
end

# # # Multiscale Stiffness and RHS