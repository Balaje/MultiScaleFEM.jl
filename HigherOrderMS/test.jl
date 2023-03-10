# include("HigherOrderMS.jl");

f(x::Float64)::Float64 = 1.0
D(x::Float64)::Float64 = (2 + cos(2π*x/2e-2))^-1

nc = 2^1
nf = 2^14
q = 1
p = 1
l = 10

quad = gausslegendre(2)
prob_data = preallocate_matrices((0.0,1.0), nc, nf, l, (q,p));

nds_coarse, elem_coarse = get_node_elem_coarse(prob_data)
nds_fine, elem_fine = get_node_elem_fine(prob_data)
coarse_indices_to_fine_indices = get_coarse_indices_to_fine_indices(prob_data)
patch_indices_to_global_indices = get_patch_indices_to_global_indices(prob_data)

stima_cache = stiffness_matrix_cache(nds_fine, elem_fine, quad, D, ∇φᵢ!, q)
lmat_cache = lm_matrix_cache((nds_coarse, nds_fine), (elem_coarse, elem_fine), quad, (q,p), Λₖ!, φᵢ!)
fvecs_cache = lm_l2_matrix_cache(nc, p)
matcache = stima_cache, lmat_cache, fvecs_cache
basis_cache = ms_basis_cache(matcache, nf, (q,p))
mat_contrib_cache, vec_contrib_cache = contrib_cache(nds_fine, coarse_indices_to_fine_indices, quad, D, f, q);

ms_elem = get_multiscale_data(prob_data)[1];
sol_cache = zeros(Float64,q*nf+1), zeros(Float64,q*nf+1);

# Compute MS bases
basis_vec_ms = compute_ms_basis(basis_cache, (q,p), l, (nf,nc), patch_indices_to_global_indices)

# Solve the problem
basis_elem_ms = BroadcastVector(getindex, Fill(basis_vec_ms,nc), coarse_indices_to_fine_indices, ms_elem);
basis_elem_ms_t = BroadcastVector(transpose, basis_elem_ms);
vec_contribs = BroadcastVector(assemble_load_vector!, vec_contrib_cache, Fill(1,nc));
mat_contribs = BroadcastVector(assemble_stiffness_matrix!, mat_contrib_cache, Fill(-1,nc));
ms_elem_mats = BroadcastVector(*, basis_elem_ms_t, mat_contribs, basis_elem_ms);
ms_elem_vecs = BroadcastVector(*, basis_elem_ms_t, vec_contribs);
stima_ms = assemble_ms_matrix(ms_elem_mats, ms_elem, nc, p);
loadvec_ms = assemble_ms_vector(ms_elem_vecs, ms_elem, nc, p);
build_solution!(sol_cache, (stima_ms\loadvec_ms), basis_vec_ms);
plot(nds_fine, sol_cache[1])