###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 

using Pkg
Pkg.activate(".")

using Gridap
using MultiscaleFEM
using SparseArrays
using ProgressMeter
using DelimitedFiles

include("./time-dependent.jl")
include("./schur.jl");

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
  p = 3;
  l = 32; # Patch size parameter
end
# f(x,t) = sin(π*x[1])*sin(π*x[2])*(sin(t))^4
f(x,t) = (sin(t))^4
u₀(x) = 0.0

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, Float64, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1);
# D(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1 # Oscillatory field
# D(x) = 1.0 # Constant field
# A = CellField(D, FineScale.trian)
# Random field
epsilon = 2^5
repeat_dims = (Int64(nf/epsilon), Int64(nf/epsilon))
a₁,b₁ = (0.5,1.5)
if(mpi_rank==0)
  rand_vals = rand(epsilon^2);
else
  rand_vals = zeros(epsilon^2);
end
MPI.Bcast!(rand_vals, 0, comm)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)
# vals_epsilon = readdlm("./coefficient.txt");
A = CellField(vec(vals_epsilon), FineScale.trian)
K = assemble_stima(V₀, A, 4);
M = assemble_massma(V₀, x->1.0, 4);

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ));
basis_vec_ms = Vₘₛ.basis_vec_ms;
Ks, Ls, Λs = Vₘₛ.fine_scale_system;

# # Compute the corrections
q = 0
L₀ = assemble_rect_matrix(Ωₘₛ, q);
Λ₀ = assemble_lm_l2_matrix(Ωₘₛ, q);
Vₘₛ′ = MultiScaleFESpace(Ωₘₛ, q, V₀, (K, L₀, Λ₀));
Wₘₛ =  MultiScaleCorrections(Vₘₛ′, p, (K, L, M, L₀));

(mpi_rank == 0) && println("Computing basis functions...")
t1 = MPI.Wtime()
B = zero(L); B₂ = zero(L₀)
build_basis_functions!((B,B₂), (Vₘₛ,Wₘₛ), comm);
t2 = MPI.Wtime()
(mpi_rank == 0) && println("Elasped time = $(t2-t1)");

if(mpi_rank == 0)
  Kₘₛ = assemble_ms_matrix(B, K);
  Mₘₛ = assemble_ms_matrix(B, M);
  Pₘₛ = assemble_ms_matrix(B, K, B₂);
  Lₘₛ = assemble_ms_matrix(B, M, B₂);
  Kₘₛ′ = assemble_ms_matrix(B₂, K);
  Mₘₛ′ = assemble_ms_matrix(B₂, M);

  global 𝐌 = [Mₘₛ Lₘₛ; 
              Lₘₛ'  Mₘₛ′];
  global 𝐊 = [Kₘₛ Pₘₛ; 
              Pₘₛ' Kₘₛ′]

  # sM = SchurComplementMatrix(𝐌, (num_cells(CoarseScale.trian)*(p+1)^2, num_cells(CoarseScale.trian)*(q+1)^2))
  # sK = SchurComplementMatrix(𝐊, (num_cells(CoarseScale.trian)*(p+1)^2, num_cells(CoarseScale.trian)*(q+1)^2))
  sM = 𝐌
  sK = 𝐊

  # Begin solving the heat equation in rank 0
  println("Solving multiscale problem...")
  function fₙ(cache, tₙ::Float64)
    Vₕ, B, B₂ = cache
    L = assemble_loadvec(Vₕ, y->f(y,tₙ), 4)
    [B'*L; B₂'*L]
  end
  Δt = 2^-7
  Δt = 2^-8
  tf = 1.0
  ntime = ceil(Int, tf/Δt)
  BDF = 4
  # Compute the reference solution with the BDFk scheme
  println("Computing reference solution ...");
  Vh = TestFESpace(Ωₘₛ.Ωf.trian, reffe, conformity=:H1, dirichlet_tags="boundary");
  Vh0 = TrialFESpace(Vh, 0.0);
  dΩ = Measure(Ωₘₛ.Ωf.trian, 5);
  a(u,v) = ∫(A*(∇(v)⊙∇(u)))dΩ;
  m(u,v) = ∫(u⊙v)dΩ;  
  Kₑ  = assemble_matrix(a, Vh0, Vh0);
  Mₑ = assemble_matrix(m, Vh0, Vh0);
  function fₙ(cache, tₙ::Float64)  
    f, Vh, dΩ = cache
    g(x) = f(x,tₙ)
    b(v) = ∫(g*v)dΩ
    assemble_vector(b, Vh)
  end
  let     
    U₀ = get_free_dof_values(interpolate(u₀, Vh0))
    global U = zero(U₀)  
    t = 0.0
    # Starting BDF steps (1...k-1) 
    fcache = (f, Vh0, dΩ) 
    for i=1:BDF-1
      dlcache = get_dl_cache(i)
      cache = dlcache, fcache
      U₁ = BDFk!(cache, t, U₀, Δt, Kₑ, Mₑ, fₙ, i)
      U₀ = hcat(U₁, U₀)
      t += Δt
    end
    # Remaining BDF steps
    dlcache = get_dl_cache(BDF)
    cache = dlcache, fcache
    @showprogress for i=BDF:ntime
      U₁ = BDFk!(cache, t+Δt, U₀, Δt, Kₑ, Mₑ, fₙ, BDF)
      U₀[:,2:BDF] = U₀[:,1:BDF-1]
      U₀[:,1] = U₁
      t += Δt
    end
    U = U₀[:,1] # Final time solution
  end
  # op = AffineFEOperator(a,b,Vh0,Vh);
  # Uex = solve(op);
  Uex = FEFunction(Vh0, U)
end

if(mpi_rank == 0)
  Kₘₛ = assemble_ms_matrix(B, K);
  Mₘₛ = assemble_ms_matrix(B, M);
  Pₘₛ = assemble_ms_matrix(B, K, B₂);
  Lₘₛ = assemble_ms_matrix(B, M, B₂);
  Kₘₛ′ = assemble_ms_matrix(B₂, K);
  Mₘₛ′ = assemble_ms_matrix(B₂, M);

  global 𝐌 = [Mₘₛ Lₘₛ; 
              Lₘₛ'  Mₘₛ′];
  global 𝐊 = [Kₘₛ Pₘₛ; 
              Pₘₛ' Kₘₛ′]

  sM = SchurComplementMatrix(𝐌, (num_cells(CoarseScale.trian)*(p+1)^2, num_cells(CoarseScale.trian)*(q+1)^2))
  sK = SchurComplementMatrix(𝐊, (num_cells(CoarseScale.trian)*(p+1)^2, num_cells(CoarseScale.trian)*(q+1)^2))
  # sM = 𝐌
  # sK = 𝐊

  # Begin solving the heat equation in rank 0
  println("Solving multiscale problem...")
  function fₙ(cache, tₙ::Float64)
    Vₕ, B, B₂ = cache
    L = assemble_loadvec(Vₕ, y->f(y,tₙ), 4)
    [B'*L; B₂'*L]
  end

  let 
    U₀ = [setup_initial_condition(u₀, B, V₀); zeros(Float64, (q+1)^2*num_cells(CoarseScale.trian))]
    global U = zero(U₀)  
    t = 0.0
    # Starting BDF steps (1...k-1) 
    fcache = (V₀, B, B₂) 
    @showprogress for i=1:BDF-1
      dlcache = get_dl_cache(i)
      cache = dlcache, fcache
      U₁ = BDFk!(cache, t, U₀, Δt, sK, sM, fₙ, i)
      U₀ = hcat(U₁, U₀)
      t += Δt
    end
    # Remaining BDF steps
    dlcache = get_dl_cache(BDF)
    cache = dlcache, fcache
    @showprogress for i=BDF:ntime
      U₁ = BDFk!(cache, t+Δt, U₀, Δt, sK, sM, fₙ, BDF)
      U₀[:,2:BDF] = U₀[:,1:BDF-1]
      U₀[:,1] = U₁
      t += Δt
    end
    U = U₀[:,1] # Final time solution
  end
  Uₘₛ = B₂*U[(p+1)^2*num_cells(CoarseScale.trian)+1:end] + B*U[1:(p+1)^2*num_cells(CoarseScale.trian)]

  Uₘₛʰ = FEFunction(Vₘₛ.Uh, Uₘₛ);  
    
  # Uex = CellField(x->sin(π*x[1])*sin(π*x[2]), FineScale.trian);
  # dΩ = Measure(get_triangulation(Vₘₛ.Uh), 4);
  # Spectrum of the matrix
  evM = eigvals(collect(Mₘₛ)); evM′ = eigvals(collect(Mₘₛ′))
  evK = eigvals(collect(Kₘₛ)); evK′ = eigvals(collect(Kₘₛ′))
  ev𝐌 = eigvals(collect(𝐌)); ev𝐊 = eigvals(collect(𝐊))
  𝐌⁻¹𝐊 = 𝐌\collect(𝐊)
  ev𝐌⁻¹𝐊 = eigvals(collect(𝐌⁻¹𝐊))

  println("Spectrum:")
  println("Mass matrix")
  println("(λₘᵢₙ, λₘₐₓ) of Mₘₛ = ($(minimum(evM)), $(maximum(evM)))")
  println("(λₘᵢₙ, λₘₐₓ) of Mₘₛ′ = ($(minimum(evM′)), $(maximum(evM′)))")
  println("(λₘᵢₙ, λₘₐₓ) of 𝐌 = ($(minimum(ev𝐌)), $(maximum(ev𝐌)))")
  println("Stiffness matrix")
  println("(λₘᵢₙ, λₘₐₓ) of Kₘₛ = ($(minimum(evK)), $(maximum(evK)))")
  println("(λₘᵢₙ, λₘₐₓ) of Kₘₛ′ = ($(minimum(evK′)), $(maximum(evK′)))")
  println("(λₘᵢₙ, λₘₐₓ) of 𝐊 = ($(minimum(ev𝐊)), $(maximum(ev𝐊)))")
  println("Maximum eigenvalue of M⁻¹K")
  println("λₘₐₓ of 𝐌⁻¹𝐊 = $(maximum(ev𝐌⁻¹𝐊))")

  L²Error = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ))
  H¹Error = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ))
  println("$L²Error   $H¹Error;")
end