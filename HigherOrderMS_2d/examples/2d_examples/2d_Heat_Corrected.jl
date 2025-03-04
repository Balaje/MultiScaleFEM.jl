###### ######## ######## ######## ######## ######### 
# Program to test the multiscale basis computation #
###### ######## ######## ######## ######## ######### 

# # Run this the first time
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Quadmath
T₁ = Float128

using Gridap
using MultiscaleFEM
using SparseArrays
using ProgressMeter
using DelimitedFiles

include("./time-dependent.jl")

domain = T₁.((0.0, 1.0, 0.0, 1.0));

##### ##### ##### ##### ##### ##### ##### 
# Temporal discretization parameters
##### ##### ##### ##### ##### ##### ##### 
Δt = 2^-7
tf = 1.0
ntime = ceil(Int, tf/Δt)
BDF = 4

##### ##### ##### ##### ##### ##### ##### 
# Spatial discretization parameters
##### ##### ##### ##### ##### ##### ##### 
(length(ARGS)==5) && begin (nf, nc, p, l, ntimes) = parse.(Int64, ARGS) end
if(length(ARGS)==0)
  nf = 2^4;
  nc = 2^1;
  p = 1;
  l = 1; # Patch size parameter
  ntimes = 1;
end

f(x,t) = T₁(2π^2*sin(π*x[1])*sin(π*x[2])*(sin(t)))
u₀(x) = T₁(0.0)

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, T₁, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1;vector_type=Vector{T₁});
# D(x) = (0.5 + 0.5*cos(2π/2^-5*x[1])*cos(2π/2^-5*x[2]))^-1 # Oscillatory field
# D(x) = 1.0 # Constant field
# A = CellField(D, FineScale.trian)

# Random field
epsilon = 2^2
repeat_dims = (Int64(nf/epsilon), Int64(nf/epsilon))
a₁,b₁ = T₁.((0.5,1.5))
rand_vals = ones(T₁,epsilon^2)
vals_epsilon = repeat(reshape(a₁ .+ (b₁-a₁)*rand_vals, (epsilon, epsilon)), inner=repeat_dims)

# vals_epsilon = readdlm("./coefficient.txt");
A = CellField(vec(vals_epsilon), FineScale.trian)

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);

# Assemble the fine scale matrices
K = assemble_stima(V₀, A, 4; T=T₁);
M = assemble_massma(V₀, x->1.0, 4; T=T₁);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

# Multiscale Space without stabilization
# γₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ)) |> collect;

# # Multiscale Space with the stabilization
Vₘₛ = MultiScaleFESpace(Ωₘₛ, p, V₀, (K, L, Λ))
γₘₛ = StabilizedMultiScaleFESpace(Vₘₛ, p, V₀, (K, L, Λ), domain, A);

# Multiscale Additional Corrections for the heat equation
Wₘₛ = Vector{MultiScaleCorrections}(undef, ntimes)
Wₘₛ[1] = MultiScaleCorrections(γₘₛ, p, (K, L, M, L)); 
for j=2:ntimes
  Wₘₛ[j] = MultiScaleCorrections(Wₘₛ[j-1], p, (K, L, M, L)); 
end

Bₘₛ = zero(L); # The bases functions
Bₘₛ′ = [zero(Bₘₛ) for j=1:ntimes]; # The additional corrections of bases functions

build_basis_functions!((Bₘₛ,), (γₘₛ,))
for j=1:ntimes
  build_basis_functions!((Bₘₛ′[j],), (Wₘₛ[j],));
end
Bₘₛ′ = hcat(Bₘₛ′...);

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the multiscale solution with the BDFk scheme
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
Kₘₛ = assemble_ms_matrix(Bₘₛ, K);
Mₘₛ = assemble_ms_matrix(Bₘₛ, M);
Pₘₛ = assemble_ms_matrix(Bₘₛ, K, Bₘₛ′);
Lₘₛ = assemble_ms_matrix(Bₘₛ, M, Bₘₛ′);
Kₘₛ′ = assemble_ms_matrix(Bₘₛ′, K);
Mₘₛ′ = assemble_ms_matrix(Bₘₛ′, M);

sM = [Mₘₛ Lₘₛ; Lₘₛ'  Mₘₛ′] |> collect;
sK = [Kₘₛ Pₘₛ; Pₘₛ' Kₘₛ′] |> collect;

println("Solving multiscale problem...")
function fₙ(cache, tₙ::Float64)
  Vₕ, B, B₂ = cache
  L = assemble_loadvec(Vₕ, y->f(y,tₙ), 4; T=T₁)
  [B'*L; B₂'*L]
end

let 
  U₀ = [setup_initial_condition(u₀, Bₘₛ, V₀; T=T₁); zeros(T₁, ntimes*(p+1)^2*num_cells(CoarseScale.trian))]
  global U = zero(U₀)  
  t = 0.0
  # Starting BDF steps (1...k-1) 
  fcache = (V₀, Bₘₛ, Bₘₛ′) 
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
Uₘₛ = Bₘₛ′*U[(p+1)^2*num_cells(CoarseScale.trian)+1:end] + Bₘₛ*U[1:(p+1)^2*num_cells(CoarseScale.trian)]
Uₘₛʰ = FEFunction(γₘₛ.Uh, Uₘₛ);    

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the reference solution with the BDFk scheme
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
println("Computing reference solution ...");
Vh = TestFESpace(Ωₘₛ.Ωf.trian, reffe, conformity=:H1, dirichlet_tags="boundary"; vector_type=Vector{T₁});
Vh0 = TrialFESpace(Vh, T₁(0.0));
dΩ = Measure(Ωₘₛ.Ωf.trian, 5; T=T₁);
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
Uex = FEFunction(Vh0, U)

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# Compute the H¹- and- L² errors using the reference solution
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
dΩ = Measure(get_triangulation(γₘₛ.Uh), 4; T=T₁);
L²Error = sqrt(sum( ∫((Uₘₛʰ - Uex)*(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫((Uex)*(Uex))dΩ ))
H¹Error = sqrt(sum( ∫(A*∇(Uₘₛʰ - Uex)⊙∇(Uₘₛʰ - Uex))dΩ ))/sqrt(sum( ∫(A*∇(Uex)⊙∇(Uex))dΩ ))
println("Done computing the solutions...")
println("$p \t $nc \t $l \t $L²Error \t $H¹Error")
