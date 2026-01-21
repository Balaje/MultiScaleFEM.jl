######### ######### ######### ######### ######### ######### ######### ######### #########
# Read the basis functions from the files and then construct the multiscale system
######### ######### ######### ######### ######### ######### ######### ######### #########

include("./fileIO.jl");
include("./time-dependent.jl")

# Load all the params
project_dir, project_name, ntimes_1 = ARGS;
param_filename = project_dir*"/"*project_name*"/$(project_name)_params.csv";
domain, nf, nc, p, l, ntimes, vals_epsilon, tf, Δt, T₁ = read_problem_parameters(param_filename);

ntimes = parse(Int64, ntimes_1)

# Define the RHS and the initial condition
f(x,t) = T₁(sin(π*x[1])*sin(π*x[2])*(sin(t))^7)
u₀(x) = T₁(0.0)
uₜ₀(x) = T₁(0.0)

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, T₁, 1);
V₀ = TestFESpace(FineScale.trian, reffe, conformity=:H1; vector_type=Vector{T₁});

A = CellField(vec(vals_epsilon), FineScale.trian)

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);

Ω = get_triangulation(V₀)
dΩ = Measure(Ω,4)

# Assemble the fine scale matrices
K = assemble_stima(V₀, A, 4; T=T₁);
M = assemble_massma(V₀, x->1.0, 4; T=T₁);
L = assemble_rect_matrix(Ωₘₛ, p);
Λ = assemble_lm_l2_matrix(Ωₘₛ, p);

function load_basis!(γₘₛ)
    @showprogress desc="Loading MS Bases..." for i=2:nc*nc
        filename = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_"*string(i)*".csv"
        γₘₛ += read_basis_functions(filename, T₁, size(L))
    end
    γₘₛ
end
function load_additional_corrections!(Wₘₛ)
    for j=1:ntimes
        @showprogress desc="Loading Additional Corrections $j..." for i=2:nc*nc
            filename = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_correction_level_$(j)_"*string(i)*".csv"
            Wₘₛ[j] += read_basis_functions(filename, T₁, size(L))
        end
    end
    Wₘₛ
end


fname_1 = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_"*string(1)*".csv"
γₘₛ = read_basis_functions(fname_1, T₁, size(L))
γₘₛ = load_basis!(γₘₛ)

fname_2(j) = project_dir*"/"*project_name*"/$(project_name)_ms_basis_$(nc)$(p)$(l)_correction_level_$(j)_"*string(1)*".csv"
Wₘₛ = [read_basis_functions(fname_2(j), T₁, size(L)) for j=1:ntimes]
Wₘₛ = load_additional_corrections!(Wₘₛ);
Wₘₛ = hcat(Wₘₛ...);

###### ###### ###### ###### ###### ###### ###### ###### ###### ######
# Compute the matrix system using the basis functions
###### ###### ###### ###### ###### ###### ###### ###### ###### ######
Kₘₛ = γₘₛ'*K*γₘₛ
Mₘₛ = γₘₛ'*M*γₘₛ
Pₘₛ = γₘₛ'*K*Wₘₛ
Lₘₛ = γₘₛ'*M*Wₘₛ
Kₘₛ′ = Wₘₛ'*K*Wₘₛ
Mₘₛ′ = Wₘₛ'*M*Wₘₛ

𝐌 = [Mₘₛ Lₘₛ; Lₘₛ' Mₘₛ′]
𝐊 = [Kₘₛ Pₘₛ; Pₘₛ' Kₘₛ′]

using OrdinaryDiffEq, OrdinaryDiffEqRKN
using IterativeSolvers, LinearMaps

ode_solver = RKN4()
solver = (y,A,b) -> y .= A\b;
M⁻¹ = InverseMap(𝐌; solver=solver);

# Define the projection of the load vector onto the multiscale space
function fₙ(cache, tₙ::Float64)
  Vₕ, B, B₂ = cache
  L = assemble_loadvec(Vₕ, y->f(y,tₙ), 8; T=T₁)
  [B'*L; B₂'*L]
end

"""
The multiscale version of the wave equation solver
"""
function W(M⁻¹::InverseMap, K::AbstractMatrix{T₁}, U₀::Vector{T₁}, 
          Uₜ₀::Vector{T₁}, V::FESpace, B₁::AbstractMatrix, B₂::AbstractMatrix,
           dt::Float64, tspan::NTuple{2, Float64})
  f_cache = V, B₁, B₂;
  p = M⁻¹, K, f_cache

  """
  The wave equation in second order form.
  """
  function W(v, u, p, t)
    M⁻¹, K, f_cache = p
    g = fₙ(f_cache, t)    
    -(M⁻¹*K*u) + M⁻¹*g
  end;

  ode_prob = SecondOrderODEProblem(W, Uₜ₀, U₀, tspan, p)
  OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt);
end;

U₀ = [setup_initial_condition(u₀, γₘₛ, V₀; T=T₁); zeros(T₁, ntimes*(p+1)^2*num_cells(CoarseScale.trian))]
Uₜ₀ = [setup_initial_condition(uₜ₀, γₘₛ, V₀; T=T₁); zeros(T₁, ntimes*(p+1)^2*num_cells(CoarseScale.trian))]

function get_sol(u)
  n = Int64(0.5*length(u))
  u[n+1:2n]
end;

tspan = (0.0,tf)
s = W(M⁻¹, 𝐊, U₀, Uₜ₀, V₀, γₘₛ, Wₘₛ, Δt, tspan);

U = get_sol(s.u[end]);

dU₁ = U[ntimes*(p+1)^2*num_cells(CoarseScale.trian)+1:end] 
U₁ = U[1:ntimes*(p+1)^2*num_cells(CoarseScale.trian)]
Uₘₛ = γₘₛ*U₁+ Wₘₛ*dU₁

using DataFrames, CSV
CSV.write(project_dir*"/"*project_name*"/$(project_name)_ms_solution_raw.csv", DataFrame((a=U)))

CSV.write(project_dir*"/"*project_name*"/$(project_name)_ms_solution.csv", DataFrame((a=Uₘₛ)))

Uref = CSV.read(project_dir*"/"*project_name*"/$(project_name)_ref_solution_$nf.csv", DataFrame, types=[T₁]).a

uₐ = FEFunction(V₀, Uₘₛ)
uₑ = FEFunction(V₀, Uref)
err = uₑ - uₐ

L²Error = sqrt(sum( ∫((err)*(err))dΩ ))
H¹Error = sqrt(sum(∫(A*(∇(err))⊙(∇(err)))dΩ))
# println("L²Error = $L²Error, \t H¹Error = $H¹Error");
println("$nf \t $nc \t $p \t $l \t $ntimes \t $L²Error \t $H¹Error")

error_data = Dict(:nf=>nf, 
                  :nc=>nc,
                  :p=>p,
                  :l=>l,
                  :ntimes=>ntimes,
                  :l2error=>L²Error,
                  :h1error=>H¹Error);
CSV.write(project_dir*"/"*project_name*"/$(project_name)_error_data.csv", DataFrame(error_data))
