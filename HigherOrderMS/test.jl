using FastGaussQuadrature
using LinearAlgebra
using Plots
using SparseArrays   
using LoopVectorization
using LazyArrays

include("preallocation.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("solve.jl")
include("multiscale_basis-functions.jl")

nc = 2^2
nf = 2^15
q = 1
p = 1
l = 10

f(x) = 1.0
D(x) = 1.0

prob_data = preallocate_matrices((0.0,1.0), nc, nf, l, (q,p), f, D, φᵢ!, ∇φᵢ!)
nds_coarse, elem_coarse = get_node_elem_coarse(prob_data)
nds_fine, elem_fine = get_node_elem_fine(prob_data)
coarse_indices_to_fine_indices = get_coarse_indices_to_fine_indices(prob_data)
quad = gausslegendre(2)

Ds, fs, Φs, ∇Φs, jacobian_exp_1, jacobian_exp_minus_1, nc_elem, p_elem, l_elem, ip_elem_wise = get_elem_wise_data(prob_data)
basis_vec_ms = get_basis_multiscale(prob_data)
patch_indices_to_global_indices = get_patch_indices_to_global_indices(prob_data)

# Compute MS bases
stima_cache = assembler_cache(nds_fine, elem_fine, quad, q)
l_mat_cache = lm_matrix_cache((nds_coarse, nds_fine), (elem_coarse, elem_fine), quad, (q,p))
lm_l2_mat_cache = lm_l2_matrix_cache(nc, p)
stima = assemble_matrix!(stima_cache, x->1.0, ∇φᵢ!, ∇φᵢ!, -1)
lmat = assemble_lm_matrix!(l_mat_cache, Λₖ!, φᵢ!, 1)
fvecs = assemble_lm_l2_matrix!(lm_l2_mat_cache, nds_coarse, elem_coarse, p)
loadvec = assemble_vector!(stima_cache, x->1.0, φᵢ!, 1)
matcache = stima, lmat, fvecs
cache = ms_basis_cache!(matcache, nf, nc, (q,p), basis_vec_ms, patch_indices_to_global_indices)
basis_vec_ms = compute_ms_bases!(cache, p, l) # Needs some more work, but managed to reduce the time substantially

cc1 = contrib_cache(nds_fine, coarse_indices_to_fine_indices, quad, q);
cc2 = contrib_cache(nds_fine, coarse_indices_to_fine_indices, quad, q);
cc1 = BroadcastVector(mat_contribs!, cc1, Ds, ∇Φs, ∇Φs, jacobian_exp_minus_1);
cc2 = BroadcastVector(vec_contribs!, cc2, fs, Φs, jacobian_exp_1);

ms_elem, sKms, sFms = get_multiscale_data(prob_data);
basis_vec_ms_el = Vector{Vector{Matrix{Float64}}}(undef,nc);
fill!(basis_vec_ms_el, basis_vec_ms);
sKms = BroadcastVector(fillsKms!, sKms, basis_vec_ms_el, coarse_indices_to_fine_indices, cc1, ip_elem_wise, nc_elem, p_elem, l_elem, 1:nc);
sFms = BroadcastVector(fillsFms!, sFms, basis_vec_ms_el, coarse_indices_to_fine_indices, cc2, ip_elem_wise, nc_elem, p_elem, l_elem, 1:nc);

stima_ms = zeros(Float64, nc*(p+1), nc*(p+1));
loadvec_ms = zeros(Float64, nc*(p+1));
assemble_ms_matrix!(stima_ms, sKms, ms_elem)
assemble_ms_vector!(loadvec_ms, sFms, ms_elem)