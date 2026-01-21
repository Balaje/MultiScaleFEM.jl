using HigherOrderMS_1d

using Gridap
using SparseArrays
using ProgressMeter
using LinearAlgebra

using OrdinaryDiffEq, OrdinaryDiffEqRKN
using IterativeSolvers, LinearMaps
using Plots

include("./time-dependent.jl")

using Random
Random.seed!(1234);

#=
Problem data
=#

T₁ = Float64

## We can use both packages for setting Quad Precision
# using Quadmath
# T₁ = Float128
# using DoubleFloats
# T₁ = Double64

domain = T₁.((0.0,1.0))
# Random diffusion coefficient
Neps = 2^8
nds_micro = LinRange(domain[1], domain[2], Neps+1)
diffusion_micro = 0.1 .+ (1-0.1)*rand(T₁,Neps+1)
function _D(x::T, nds_micro::AbstractVector{T}, diffusion_micro::Vector{T1}) where {T<:Number, T1<:Number}
  n = size(nds_micro, 1)
  for i=1:n
    if(nds_micro[i] < x < nds_micro[i+1])      
      return diffusion_micro[i+1]
    elseif(x==nds_micro[i])
      return diffusion_micro[i+1]
    elseif(x==nds_micro[i+1])
      return diffusion_micro[i+1]
    else
      continue
    end 
  end
end
A(x; nds_micro = nds_micro, diffusion_micro = diffusion_micro) = _D(x[1], nds_micro, diffusion_micro)
# A(x) = 0.45;
# f(x,t) = (x[1]<0.5) ? T₁(0.0) : T₁(sin(π*x[1])*(sin(t))^5)
f(x,t) = T₁(sin(π*x[1])*sin(t)^7)
u₀(x) = T₁(0.0)
uₜ₀(x) = T₁(0.0)

ode_solver = RKN4()

# Spatial discretization parameters
# (length(ARGS)==5) && begin (nf, nc, p, l, ntimes) = parse.(Int64, ARGS) end
if(length(ARGS)==0)
  nf = 2^11;
  p = 1;
  nc = 2^4;  
  l = nc; 
  ntimes = 1;
else
  (nf, nc, p, l, ntimes) = parse.(Int64, ARGS)
end

# Temporal discretization parameters
tf = 1.0

# Solve the fine scale problem onfce for exact solution
model = CartesianDiscreteModel(domain, (nf,));
Ω = Triangulation(model);
dΩ = Measure(Ω, 2);

Uₕ = TestFESpace(model, ReferenceFE(lagrangian, T₁, 1), conformity=:H1, dirichlet_tags="boundary", vector_type=Vector{T₁}); # Test Space
Uₕ₀ = TrialFESpace(Uₕ, 0.0); # Trial Space

aₕ(u,v) = ∫(A*(∇(u)⋅∇(v)))dΩ;
mₕ(u,v) = ∫(u*v)dΩ;
function lₕ(t,v)
  g(x) = f(x,t)
  ∫(g*v)dΩ;
end

K = assemble_matrix(aₕ, Uₕ₀, Uₕ);
M = assemble_matrix(mₕ, Uₕ₀, Uₕ);
# solver = (y,A,b) -> minres!(fill!(y,0.0), A, b; reltol=eps(T₁), abstol=eps(T₁));
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

tspan = (0.0,tf);

U₀ = M⁻¹*assemble_vector(v->∫(u₀*v)dΩ, Uₕ₀);
Uₜ₀ = M⁻¹*assemble_vector(v->∫(uₜ₀*v)dΩ, Uₕ₀);

# println("\nSolving the Reference Problem:\n")
# Δt = 2^-6
# println("Trying to solve using Δt = $Δt.")
# s = W(M⁻¹, K, U₀, Uₜ₀, Uₕ, Uₕ₀, Δt, tspan);

Δt = 2^-12
# println("Trying to solve using Δt = $Δt.")
s = W(M⁻¹, K, U₀, Uₜ₀, Uₕ, Uₕ₀, Δt, tspan);

Uex = get_sol(s.u[end]);
uₕ = FEFunction(Uₕ₀, Uex);

###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
# Begin solving using the new multiscale method and compare the convergence rates #
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
fine_scale_space = FineScaleSpace(domain, 1, 6, nf; T=T₁);

stima = assemble_stiffness_matrix(fine_scale_space, A);
massma = assemble_mass_matrix(fine_scale_space, x->1);

# Define the projection of the load vector onto the multiscale space
function fₙ!(cache, tₙ::Float64)
  # "A Computationally Efficient Method"
  fspace, basis_vec_ms, basis_vec_ms₂ = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  [basis_vec_ms₂'*loadvec; basis_vec_ms'*loadvec]
end   

# Obtain the map between the coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (1,p));

# Obtain the basis functions
basis_vec_ms₁ = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices; T=T₁);

# Compute the stabilized basis functions
γ = compute_stabilized_ms_basis(fine_scale_space, A, p, nc, l; T=T₁);
basis_vec_ms₁[:, 1:(p+1):(p+1)*nc] = γ;    

# Compute the additional correction basis
basis_vec_ms₂ = compute_additional_correction_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices, p, basis_vec_ms₁; T=T₁, ntimes=ntimes);      

# Assemble the stiffness, mass matrices
Kₘₛ = basis_vec_ms₁'*stima*basis_vec_ms₁; 
Mₘₛ = basis_vec_ms₁'*massma*basis_vec_ms₁; 
Kₘₛ′ = basis_vec_ms₂'*stima*basis_vec_ms₂; 
Mₘₛ′ = basis_vec_ms₂'*massma*basis_vec_ms₂; 
Lₘₛ = basis_vec_ms₂'*massma*basis_vec_ms₁
Pₘₛ = basis_vec_ms₂'*stima*basis_vec_ms₁

𝐌 = [Mₘₛ′ Lₘₛ; Lₘₛ'  Mₘₛ];
𝐊 = [Kₘₛ′ Pₘₛ; Pₘₛ' Kₘₛ];

𝐌⁻¹ = InverseMap(𝐌; solver=solver);

"""
The multiscale version of the wave equation solver
"""
function Wₘₛ(M⁻¹::InverseMap, K::AbstractMatrix{T₁}, U₀::Vector{T₁}, 
          Uₜ₀::Vector{T₁}, V::FineScaleSpace, B₁::AbstractMatrix, B₂::AbstractMatrix,
           dt::Float64, tspan::NTuple{2, Float64})
  f_cache = V, B₁, B₂;
  p = M⁻¹, K, f_cache

  """
  The wave equation in second order form.
  """
  function W(v, u, p, t)
    M⁻¹, K, f_cache = p
    g = fₙ!(f_cache, t)    
    -(M⁻¹*K*u) + M⁻¹*g
  end;

  ode_prob = SecondOrderODEProblem(W, Uₜ₀, U₀, tspan, p)
  OrdinaryDiffEq.solve(ode_prob, ode_solver, dt = dt);
end;

U₀ = 𝐌⁻¹*[basis_vec_ms₂'*(zeros(T₁, num_free_dofs(fine_scale_space.U))); 
           basis_vec_ms₁'*(assemble_vector(v->∫(u₀*v)fine_scale_space.dΩ, fine_scale_space.U))];
Uₜ₀ = 𝐌⁻¹*[basis_vec_ms₂'*(zeros(T₁, num_free_dofs(fine_scale_space.U))); 
           basis_vec_ms₁'*(assemble_vector(v->∫(uₜ₀*v)fine_scale_space.dΩ, fine_scale_space.U))];

# println("\nSolving the Multiscale Problem:\n")
# Δt = 2^-4;
# println("Trying to solve using Δt = $Δt.")
s = Wₘₛ(𝐌⁻¹, 𝐊, U₀, Uₜ₀, fine_scale_space, basis_vec_ms₁, basis_vec_ms₂, Δt, tspan);

U = get_sol(s.u[end])
# Construct the corrected solution
U₁ = U[ntimes*(p+1)*nc+1:end] 
dU₁ = U[1:ntimes*(p+1)*nc]
U_fine_scale = basis_vec_ms₁*U₁+ basis_vec_ms₂*dU₁

# Compute the errors
uₘₛ = FEFunction(fine_scale_space.U, U_fine_scale);
σₖ = get_cell_dof_ids(Uₕ₀);
mᵦ = Broadcasting(Gridap.Arrays.PosNegReindex(Uex, [T₁(0.0), T₁(0.0)]))
uₕ = CellField(fine_scale_space.U, lazy_map(mᵦ, σₖ));
e = uₕ - uₘₛ;
L²Error = sqrt(sum(∫(e*e)fine_scale_space.dΩ));
H¹Error = sqrt(sum(∫(∇(e)⋅∇(e))fine_scale_space.dΩ));

println("")
println("(1/h) \t (1/H) \t p \t l \t j \t ||⋅||₀ \t √(a(⋅,⋅))")
println("")
println("$nf \t $nc \t $p \t $l \t $ntimes \t $L²Error \t $H¹Error")