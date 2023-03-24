include("HigherOrderMS.jl");

domain = (0.0,1.0)

D(x) = 1.0
f(x) = π^2*cos(π*x[1])

nc = 2^5
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
plt3 = plot(nds_fine, sol_fine_scale, label="Homogeneous DBC")

# Non-homogeneous boundary condition [1.0,1.0]
fullnodes = 1:q*nf+1;
bnodes = [1,q*nf+1];
bvals = [1.0,-1.0];
freenodes = setdiff(fullnodes, bnodes);
# "The boundary correction term" - needs to be computed once
boundary_correction = (stima[freenodes,freenodes]\collect(stima[freenodes, bnodes]));
sol_fine_scale_dbc = basis_vec_ms[freenodes,:]*sol - boundary_correction*bvals;
plt4 = plot(nds_fine[freenodes], sol_fine_scale_dbc, label="Non-homogeneous DBC");