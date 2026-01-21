# using Quadmath
using Gridap
using SparseArrays
using ProgressMeter


using OrdinaryDiffEq, OrdinaryDiffEqRKN
using LinearMaps

include("./fileIO.jl")
include("./time-dependent.jl")

project_dir, project_name = ARGS
param_filename = project_dir*"/"*project_name*"/$(project_name)_params.csv";
domain, nf, nc, p, l, ntimes, vals_epsilon, tf, Δt, T₁ = read_problem_parameters(param_filename);

##### ##### ##### ##### ##### ##### #####
# Temporal discretization scheme
##### ##### ##### ##### ##### ##### #####

f(x,t) = T₁(sin(π*x[1])*sin(π*x[2])*(sin(t))^7)
u₀(x) = T₁(0.0)
uₜ₀(x) = T₁(0.0)

##### ##### ##### ##### ##### #####
# Compute the reference solution
##### ##### ##### ##### ##### #####
model = CartesianDiscreteModel(domain, (nf, nf))
Ω = Triangulation(model)
dΩ = Measure(Ω, 5; T=T₁)
A = CellField(vec(vals_epsilon), Ω)
reffe = ReferenceFE(lagrangian, T₁, 1)
Uₕ = TestFESpace(Ω, reffe, conformity=:H1, dirichlet_tags="boundary"; vector_type=Vector{T₁});
Uₕ₀ = TrialFESpace(Uₕ, T₁(0.0));
a(u,v) = ∫(A*(∇(v)⊙∇(u)))dΩ;
m(u,v) = ∫(u⊙v)dΩ;
K = assemble_matrix(a, Uₕ₀, Uₕ);
M = assemble_matrix(m, Uₕ₀, Uₕ);
function lₕ(t,v)
  g(x) = f(x,t)
  ∫(g*v)dΩ;
end

ode_solver=RKN4()
solver = (y,A,b) -> y .= A\b;
M⁻¹ = InverseMap(M; solver=solver);

"""
Solver function for the wave equation
"""
function W(M⁻¹::InverseMap, K::AbstractMatrix{T₁}, U₀::Vector{T₁}, 
          Uₜ₀::Vector{T₁}, U::FESpace, U0::FESpace, dt::Float64, tspan::NTuple{2, Float64})
  p = M⁻¹, K, U, U0

  """
  The wave equation in second order form.
  """
  function W(v, u, p, t)
    M⁻¹, K, V, V0 = p
    g = assemble_vector(v->lₕ(t,v), V0)    
    -(M⁻¹*K*u) + M⁻¹*g
  end;

  ode_prob = SecondOrderODEProblem(W, Uₜ₀, U₀, tspan, p)
  OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt);
end;

function get_sol(u)
  n = Int64(0.5*length(u))
  u[n+1:2n]
end;

U₀ = M⁻¹*assemble_vector(v->∫(u₀*v)dΩ, Uₕ₀);
Uₜ₀ = M⁻¹*assemble_vector(v->∫(uₜ₀*v)dΩ, Uₕ₀);

tspan = (0.0,tf);
s = W(M⁻¹, K, U₀, Uₜ₀, Uₕ, Uₕ₀, Δt, tspan);
U = get_sol(s.u[end]);

V₀ = TestFESpace(Ω, reffe, conformity=:H1; vector_type=Vector{T₁});
uₕ = interpolate_everywhere(FEFunction(Uₕ, U), V₀);

using DataFrames, CSV
CSV.write(project_dir*"/"*project_name*"/$(project_name)_ref_solution_$nf.csv", DataFrame((a=uₕ |> get_free_dof_values)))
