include("HigherOrderMS.jl")

domain = (0.0, 1.0)

# c²(x) = 4.0
c²(x) = (0.25 + 0.125*cos(2π*x[1]/2e-2))^-1
f(x,t) = 0.0
u₀(x) = 0.0
uₜ₀(x) = 4π*sin(2π*x[1])
u(x,t) = sin(2π*x)*sin(4π*t)

# Problem parameters
nc = 2^4
nf = 2^16
p = 1
q = 1
l = 7
qorder = 4

# Get the Gridap fine-scale description
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)
nds_fine = LinRange(domain[1], domain[2], q*nf+1)

# Compute the map between the coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
# Compute the multiscale basis
basis_vec_ms = compute_ms_basis(fine_scale_space, c², p, nc, l, patch_indices_to_global_indices)
# Assemble the stiffness, mass matrices and define the load vector as a function of time.
stima = assemble_stiffness_matrix(fine_scale_space, c²)
massma = assemble_mass_matrix(fine_scale_space, x->1.0)
Kₘₛ = basis_vec_ms'*stima*basis_vec_ms
Mₘₛ = basis_vec_ms'*massma*basis_vec_ms
function fₙ!(cache, tₙ::Float64)
  fspace, basis_vec_ms = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  basis_vec_ms'*loadvec
end
Δt = 10^-3
tf = 1.125
ntime = ceil(Int, tf/Δt)
let 
  U₀ = setup_initial_condition(u₀, basis_vec_ms, fine_scale_space)
  V₀ = setup_initial_condition(uₜ₀, basis_vec_ms, fine_scale_space)
  global U = zero(U₀)
  cache = fine_scale_space, basis_vec_ms
  t = 0.0
  for i=1:ntime
    U₁, V₁ = CN!(cache, t, U₀, V₀, Δt, Kₘₛ, Mₘₛ, fₙ!)
    U₀, V₀ = U₁, V₁
    t += Δt
  end
  U = U₀ # Final time solution  
end
U_fine_scale = basis_vec_ms*U

# Plot
plt1 = plot(nds_fine, U_fine_scale, label="Approx. Solution using MS Method", lc=:blue, lw=2)
plot!(plt1, nds_fine, [u(x, tf) for x in nds_fine], label="Exact solution", lc=:red, lw=1)