# using Quadmath
using Gridap
using SparseArrays
using ProgressMeter

include("./fileIO.jl")
include("./time-dependent.jl")

project_dir, project_name = ARGS
param_filename = project_dir*"/"*project_name*"/$(project_name)_params.csv";
domain, nf, nc, p, l, ntimes, vals_epsilon, tf, Δt, T₁ = read_problem_parameters(param_filename);

##### ##### ##### ##### ##### ##### #####
# Temporal discretization scheme
##### ##### ##### ##### ##### ##### #####
ntime = ceil(Int, tf/Δt)
BDF = 4

f(x,t) = T₁(10*2π^2*sin(π*x[1])*sin(π*x[2])*(sin(t))^5)
u₀(x) = T₁(0.0)

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Compute the reference solution with the BDFk scheme
##### ##### ##### ##### ##### ##### ##### ##### ##### #####
println("Computing reference solution ...");
model = CartesianDiscreteModel(domain, (nf, nf))
Ω = Triangulation(model)
dΩ = Measure(Ω, 5; T=T₁)
A = CellField(vec(vals_epsilon), Ω)
reffe = ReferenceFE(lagrangian, T₁, 1)
Vh = TestFESpace(Ω, reffe, conformity=:H1, dirichlet_tags="boundary"; vector_type=Vector{T₁});
Vh0 = TrialFESpace(Vh, T₁(0.0));
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
  @showprogress for i=1:BDF-1
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

Vh1 = TestFESpace(Ω, reffe, conformity=:H1; vector_type=Vector{T₁});
Uₕ = FEFunction(Vh0, U);
Uₕ = interpolate_everywhere(Uₕ, Vh1);

using DataFrames, CSV
CSV.write(project_dir*"/"*project_name*"/$(project_name)_ref_solution_$nf.csv", DataFrame((a=Uₕ |> get_free_dof_values)))
