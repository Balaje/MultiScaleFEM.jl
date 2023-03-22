include("HigherOrderMS.jl")

domain = (0.0,1.0)

A(x) = 0.5
f(x,t) = 0.0
u₀(x) = sin(π*x[1])

# Problem parameters
nc = 2^4
nf = 2^16
p = 1
q = 1
l = 7
qorder = 4

# Get the Gridap fine-scale description
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)

# Compute the map between the coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
# Compute the multiscale basis
basis_vec_ms = compute_ms_basis(fine_scale_space, A, p, nc, l, patch_indices_to_global_indices);
# Assemble the stiffness, mass matrices and define the load vector as a function of time.
stima = assemble_stiffness_matrix(fine_scale_space, A)
massma = assemble_mass_matrix(fine_scale_space, x->1.0)
Kₘₛ = basis_vec_ms'*stima*basis_vec_ms
Mₘₛ = basis_vec_ms'*massma*basis_vec_ms
function fₙ!(cache, tₙ::Float64)
  fspace, basis_vec_ms = cache
  loadvec = assemble_load_vector(fspace, y->f(y,tₙ))
  basis_vec_ms'*loadvec
end
Δt = 10^-3
tf = 1.0
ntime = ceil(Int, tf/Δt)
BDF = 4
let 
  U₀ = setup_initial_condition(u₀, basis_vec_ms, fine_scale_space)  
  global U = zero(U₀)  
  t = 0.0
  # Starting BDF steps (1...k-1) 
  fcache = (fine_scale_space, basis_vec_ms) 
  for i=1:BDF-1
    dlcache = get_dl_cache(i)
    cache = dlcache, fcache
    U₁ = BDFk!(cache, t, U₀, Δt, Kₘₛ, Mₘₛ, fₙ!, i)
    U₀ = hcat(U₁, U₀)
    t += Δt
  end
  # Remaining BDF steps
  dlcache = get_dl_cache(BDF)
  cache = dlcache, fcache
  for i=BDF:ntime
    U₁ = BDFk!(cache, t+Δt, U₀, Δt, Kₘₛ, Mₘₛ, fₙ!, BDF)
    U₀[:,2:BDF] = U₀[:,1:BDF-1]
    U₀[:,1] = U₁
    t += Δt
  end
  U = U₀[:,1] # Final time solution
end
U_fine_scale = basis_vec_ms*U

# Plot
nds_fine = LinRange(domain[1], domain[2], q*nf+1)
plt2 = plot(nds_fine, U_fine_scale)
plot!(plt2, nds_fine, exp(-0.5*π^2*tf)*u₀.(nds_fine))