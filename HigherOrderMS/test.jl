using FastGaussQuadrature
using LinearAlgebra
using Plots

include("preallocation.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("solve.jl")
include("multiscale_basis-functions.jl")

nc = 2^2
nf = 2^16
q = 1
p = 1

prob_data = PreAllocateMatrices.preallocate_matrices((0.0,1.0), nc, nf, 2, (q,p))
nds_coarse = prob_data[1][1]
elem_coarse = prob_data[1][2]
nds_fine = prob_data[1][3]
elem_fine = prob_data[1][4]
quad = gausslegendre(4)


stima_cache = AssembleMatrices.assembler_cache(nds_fine, elem_fine, quad, q)
l_mat_cache = AssembleMatrices.lm_matrix_cache((nds_coarse, nds_fine), (elem_coarse, elem_fine), quad, (q,p))
lm_l2_mat_cache= AssembleMatrices.lm_l2_matrix_cache(nds_coarse, elem_coarse, p, quad)

matcache = stima_cache, l_mat_cache, lm_l2_mat_cache
cache = MultiScaleBases.ms_basis_cache!(matcache, (nds_coarse, nds_fine), (elem_coarse, elem_fine), quad, (q,p), x->1.0, prob_data[6], prob_data[3])