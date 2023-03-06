using FastGaussQuadrature
using LinearAlgebra
using Plots
using SparseArrays   
using LoopVectorization

include("preallocation.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("solve.jl")
include("multiscale_basis-functions.jl")

nc = 2^1
nf = 2^8
q = 1
p = 1
l = 1

prob_data = preallocate_matrices((0.0,1.0), nc, nf, l, (q,p))
nds_coarse, elem_coarse = get_node_elem_coarse(prob_data)
nds_fine, elem_fine = get_node_elem_fine(prob_data)
coarse_indices_to_fine_indices = get_coarse_indices_to_fine_indices(prob_data)
quad = gausslegendre(2)

basis_vec_ms = get_basis_multiscale(prob_data)
patch_indices_to_global_indices = get_patch_indices_to_global_indices(prob_data)

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

cc = contrib_cache(nds_fine, coarse_indices_to_fine_indices, quad, q)
mat_contribs!(cc, x->1.0,  ∇φᵢ!, ∇φᵢ!, -1)
vec_contribs!(cc, x->1.0, φᵢ!, 1)

ms_elem, sKms, sFms = get_multiscale_data(prob_data)
L, Lt, ipcache = get_innerproduct_cache(prob_data)
cache = sKms, basis_vec_ms, coarse_indices_to_fine_indices, cc, (L,Lt,ipcache)
sKms = fillsKms!(cache, nc, p, l)
sFms = fillsFms!(cache, nc, p, l)