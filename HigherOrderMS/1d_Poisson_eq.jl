include("HigherOrderMS.jl");

domain = (0.0,1.0)

D(x) = (2 + cos(2π*x[1]/2e-2))^-1
f(x) = 1.0

nc = 2^1
nf = 2^16
q = 1
p = 1
l = 4
qorder = 2

# Get the Gridap fine-scale description
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)

# Compute the map between the coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
# Compute Multiscale bases
basis_vec_ms = compute_ms_basis(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices);
# Solve the problem
stima = assemble_stiffness_matrix(fine_scale_space, D)
loadvec = assemble_load_vector(fine_scale_space, f)
Kₘₛ = basis_vec_ms'*stima*basis_vec_ms;
Fₘₛ = basis_vec_ms'*loadvec;
sol = Kₘₛ\Fₘₛ

# Plot
nds_fine = LinRange(domain[1], domain[2], q*nf+1)
sol_fine_scale = basis_vec_ms*sol
plt3 = plot(nds_fine, sol_fine_scale)

# Non-homogeneous boundary conditions
fullnodes = 1:q*nf+1
bnodes = [1,q*nf+1]
freenodes = setdiff(fullnodes, bnodes)
bvals = [1.0,1.0]
gₕ = zeros(Float64, q*nf+1)
gₕ[bnodes] = bvals

basis_vec_corr = compute_ms_correctors(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices, gₕ)
basis_vec = zero(basis_vec_ms)
basis_vec[:, (p+2):(p+1)*nc-(p+2), :] = basis_vec_ms[:,(p+2):(p+1)*nc-(p+2)]
basis_vec[:, vcat(1:(p+1), (p+1)*nc-(p):(p+1)*nc)] = basis_vec_corr;

Kₘₛ1 = basis_vec'*stima*basis_vec;
Fₘₛ1 = basis_vec'*loadvec;
sol1 = Kₘₛ1\Fₘₛ1
plot(nds_fine, basis_vec*sol1)