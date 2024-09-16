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

domain = (0.0, 1.0, 0.0, 1.0);

# Fine scale space description
# nf, nc, p, l = parse.(Int64, ARGS)
# if(ARGS == Nothing)
  nf = 2^7;
  nc = 2^3;
  p = 2;
  l = 5; # Patch size parameter
# end
# A(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1
A(x) = 1.0
f(x) = sin(3π*x[1])*sin(5π*x[2])

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, Float64, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1);
K = assemble_stima(V₀, A, 0);
F = assemble_loadvec(V₀, f, 4);

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
basis_vec_ms, patch_interior_fine_scale_dofs, coarse_dofs = Vₘₛ.basis_vec_ms;
Ks, Ls, Λs = Vₘₛ.fine_scale_system;
# Lazy fill the fine-scale vector
Fs = lazy_fill(F, num_cells(CoarseScale.trian));

t1 = MPI.Wtime()
mpi_rank = MPI.Comm_rank(comm);
n_cells_per_proc = Int64(num_cells(CoarseScale.trian)/mpi_size)
B = zeros(Float64, size(L,1), n_cells_per_proc*(p+1)^2)
@showprogress for i=n_cells_per_proc*(mpi_rank)+1:n_cells_per_proc*(mpi_rank+1)  
  B[patch_interior_fine_scale_dofs[i], coarse_dofs[i] .- mpi_rank*n_cells_per_proc*(p+1)^2] = basis_vec_ms[i];  
end
B = MPI.Gather(B, comm);
t2 = MPI.Wtime()
(mpi_rank == 0) && println("Elasped time = $(t2-t1)");

if(mpi_rank == 0)
  B = reshape(B, size(L)...)
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