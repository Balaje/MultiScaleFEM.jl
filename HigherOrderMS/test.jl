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

nc = 2^8
nf = 2^16
q = 1
p = 1
l = 10

f(x) = 1.0
D(x) = 1.0

prob_data = preallocate_matrices((0.0,1.0), nc, nf, l, (q,p), f, D, φᵢ!, ∇φᵢ!);

nds_coarse, elem_coarse = get_node_elem_coarse(prob_data)
nds_fine, elem_fine = get_node_elem_fine(prob_data)
coarse_indices_to_fine_indices = get_coarse_indices_to_fine_indices(prob_data)
quad = gausslegendre(2)

# Extract all the caches
Ds, fs, Φs, ∇Φs, jacobian_exp, nc_elem, p_elem, l_elem, ip_elem_wise = get_elem_wise_data(prob_data)
basis_vec_ms = get_basis_multiscale(prob_data)
patch_indices_to_global_indices = get_patch_indices_to_global_indices(prob_data)
stima_cache = stiffness_matrix_cache(nds_fine, elem_fine, quad, q)
lmat_cache = lm_matrix_cache((nds_coarse, nds_fine), (elem_coarse, elem_fine), quad, (q,p))
fvecs_cache = lm_l2_matrix_cache(nc, p)
matcache = stima_cache, lmat_cache, fvecs_cache
basis_cache = ms_basis_cache!(matcache, D, nf, (q,p), basis_vec_ms[1], patch_indices_to_global_indices)
mat_contrib_cache, vec_contrib_cache = contrib_cache(nds_fine, coarse_indices_to_fine_indices, quad, q);
ms_elem = get_multiscale_data(prob_data)[1];
stima_ms = zeros(Float64, nc*(p+1), nc*(p+1));
loadvec_ms = zeros(Float64, nc*(p+1));
sol_cache = zeros(Float64,q*nf+1), zeros(Float64,q*nf+1);

# Compute MS bases
compute_ms_bases!(basis_cache, p, l)
# Function to sort the basis vectors element-wise in the coarse scale
basis_elem_ms = sort_basis_vectors!(basis_vec_ms[2], basis_vec_ms[1], ms_elem, p, l)

mat_contribs = BroadcastVector(assemble_stiffness_matrix!, mat_contrib_cache, Ds, ∇Φs, ∇Φs, jacobian_exp*(-1));
ms_elem_mats = BroadcastVector(fillsKms!, ip_elem_wise, basis_elem_ms, coarse_indices_to_fine_indices, mat_contribs);
assemble_ms_matrix!(stima_ms, ms_elem_mats, ms_elem);

# fs = convert_to_cell_wise(f, nc);
vec_contribs = BroadcastVector(assemble_load_vector!, vec_contrib_cache, fs, Φs, jacobian_exp);
ms_elem_vecs = BroadcastVector(fillsFms!, ip_elem_wise, basis_elem_ms, coarse_indices_to_fine_indices, vec_contribs);
assemble_ms_vector!(loadvec_ms, ms_elem_vecs, ms_elem);  
build_solution!(sol_cache, (stima_ms\loadvec_ms), basis_vec_ms[1]);


# # Lazy way
# function benchmark_lazy_implementation()
#   mat_contribs = BroadcastVector(assemble_stiffness_matrix!, mat_contrib_cache, Ds, ∇Φs, ∇Φs, jacobian_exp*(-1));
#   sKms = BroadcastVector(fillsKms, basis_vec_ms_el, coarse_indices_to_fine_indices, mat_contribs, ip_elem_wise, nc_elem, p_elem, l_elem, 1:nc);
#   assemble_ms_matrix!(stima_ms, sKms, ms_elem);
#   for i=1:1000
#     fs = convert_to_cell_wise(f, nc);
#     vec_contribs = BroadcastVector(assemble_load_vector!, vec_contrib_cache, fs, Φs, jacobian_exp);
#     sFms = BroadcastVector(fillsFms, basis_vec_ms_el, coarse_indices_to_fine_indices, vec_contribs, ip_elem_wise, nc_elem, p_elem, l_elem, (1:nc));
#     assemble_ms_vector!(loadvec_ms, sFms, ms_elem); 
#     build_solution!(sol_cache, (stima_ms\loadvec_ms), basis_vec_ms);
#   end
# end