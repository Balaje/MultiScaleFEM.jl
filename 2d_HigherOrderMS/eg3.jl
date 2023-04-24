###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^7
q = 1
nc = 2^4
p = 1
l = 1; # Patch size parameter
ms_space = MultiScaleFESpace(domain, q, p, nf, nc, l)

num_coarse_cells = num_cells(get_triangulation(ms_space.UH))
num_fine_cells = num_cells(get_triangulation(ms_space.Uh))
node_coordinates = get_node_coordinates(get_triangulation(ms_space.Uh))

coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells)

patch_coarse_models = get_patch_triangulation(ms_space, num_coarse_cells, CoarseScale())
patch_fine_models = get_patch_triangulation(ms_space, num_coarse_cells, FineScale())
elem_wise_models = get_patch_triangulation(ms_space, num_coarse_cells, FineScaleElemWise())

patch_fine_spaces = lazy_map(build_patch_fine_spaces, patch_fine_models, Gridap.Arrays.Fill(q, num_fine_cells));
patch_coarse_trian = lazy_map(Triangulation, patch_coarse_models)

A(x) = 1.0
patch_stima = lazy_map(assemble_stima, patch_fine_spaces, Gridap.Arrays.Fill(A, num_coarse_cells), Gridap.Arrays.Fill(4, num_coarse_cells));

patch_local_node_ids = get_elem_data(ms_space)[3]
# patch_lmat = lazy_map(assemble_rect_matrix, patch_coarse_trian, patch_fine_spaces, Gridap.Arrays.Fill(p,num_coarse_cells), patch_local_node_ids)
