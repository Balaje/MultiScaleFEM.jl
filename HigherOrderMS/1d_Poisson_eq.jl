include("HigherOrderMS.jl");

domain = (0.0,1.0)

D(x) = 1.0
f(x) = π^2*sin(π*x[1])

nc = 2^4
nf = 2^15
q = 1
p = 1
l = 7
qorder = 2

nds_fine = LinRange(domain[1], domain[2], q*nf+1)

# Get the Gridap fine-scale description
fine_scale_space = FineScaleSpace(domain, q, qorder, nf)

# Compute the map between the coarse and fine scale
patch_indices_to_global_indices, coarse_indices_to_fine_indices, ms_elem = coarse_space_to_fine_space(nc, nf, l, (q,p));
# Compute Multiscale bases
basis_vec_ms = compute_ms_basis(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices);
# Construct the full stiffness and load vectors
stima = assemble_stiffness_matrix(fine_scale_space, D)
loadvec = assemble_load_vector(fine_scale_space, f)

# Solve the problem after constructing the stiffness and load vectors
Kₘₛ = basis_vec_ms'*stima*basis_vec_ms;
Fₘₛ = basis_vec_ms'*loadvec;
sol = Kₘₛ\Fₘₛ
# Plot
sol_fine_scale = basis_vec_ms*sol
plt3 = plot(nds_fine, sol_fine_scale, label="Homogeneous DBC")

# -- Example with non-homogenous Dirichlet boundary conditions
f(x) = π^2*cos(π*x[1])
ug(x) = cos(π*x[1])
# First method for Non-homogeneous boundary condition
loadvec = assemble_load_vector(fine_scale_space, f)
fullnodes = 1:q*nf+1;
bnodes = [1,q*nf+1];
bvals = ug.(nds_fine[bnodes]);
freenodes = setdiff(fullnodes, bnodes);
Kₘₛ = basis_vec_ms'*stima*basis_vec_ms;
Fₘₛ = basis_vec_ms'*loadvec;
sol = Kₘₛ\Fₘₛ;
# "The boundary correction term". Needs to be computed once by inverting the full stiffness matrix
boundary_correction = (stima[freenodes,freenodes]\collect(stima[freenodes, bnodes]));
sol_fine_scale_dbc = zeros(Float64, q*nf+1)
sol_fine_scale_dbc[freenodes] = basis_vec_ms[freenodes,:]*sol - boundary_correction*bvals;
sol_fine_scale_dbc[bnodes] = bvals
# Plot
plt4 = plot(nds_fine, sol_fine_scale_dbc, label="Non-homogeneous DBC");

#= 
However inverting the whole stiffness matrix in Line 40 may not always be feasible.
The remedy is to incorporate the BC using only the patch problems on the boundary 
(See [Henning, Målqvist 2018](https://epubs.siam.org/doi/10.1137/130933198))
=#
# Second Method for non-homogeneous boundary condition
fullnodes = 1:q*nf+1;
bnodes = [1,q*nf+1];
bvals = bvals = ug.(nds_fine[bnodes]);
freenodes = setdiff(fullnodes, bnodes);
# Obtain the matrix which acts as the corrector. 
# This involves inverting 2 small matrices on the patch of the boundary elements
Pₕug = compute_boundary_correction_matrix(fine_scale_space, D, p, nc, l, patch_indices_to_global_indices);
# Obtain the boundary contribution by multiplying the matrix with the Dirichlet boundary
boundary_contrib = apply_boundary_correction(Pₕug, bnodes, bvals, patch_indices_to_global_indices, p, nc, l, fine_scale_space);
# Solve the interior problem
Kₘₛ = basis_vec_ms'*stima[:,freenodes]*basis_vec_ms[freenodes,:];
Fₘₛ = basis_vec_ms'*loadvec - basis_vec_ms'*(stima[:,bnodes]*bvals);
sol2 = Kₘₛ\Fₘₛ;
# Apply the boundary correction
sol_fine_scale_dbc_2 = basis_vec_ms*sol2 + boundary_contrib
# Plot
plt5 = plot(nds_fine, sol_fine_scale_dbc_2, label="Non-homogeneous DBC");

# Compute the error using the reference solution
sol_ref = zeros(Float64, q*nf+1)
sol_ref[freenodes] = stima[freenodes,freenodes]\(loadvec[freenodes] - stima[freenodes,bnodes]*bvals)
sol_ref[bnodes] = bvals

uₕ = FEFunction(fine_scale_space.U, sol_ref)
u₁ = FEFunction(fine_scale_space.U, sol_fine_scale_dbc)
u₂ = FEFunction(fine_scale_space.U, sol_fine_scale_dbc_2)

dΩ = fine_scale_space.dΩ  
l²e₁ = sqrt(sum(∫((u₁-uₕ)*(u₁-uₕ))dΩ))
h¹e₁ = sqrt(sum(∫(∇(u₁-uₕ)⋅∇(u₁-uₕ))dΩ))
@show l²e₁, h¹e₁
l²e₂ = sqrt(sum(∫((u₂-uₕ)*(u₂-uₕ))dΩ))
h¹e₂ = sqrt(sum(∫(∇(u₂-uₕ)⋅∇(u₂-uₕ))dΩ))
@show l²e₂ , h¹e₂