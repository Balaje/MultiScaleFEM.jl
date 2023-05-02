###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^4
q = 1
nc = 2^2
p = 1
l = 1 # Patch size parameter
ms_space = MultiScaleFESpace(domain, q, p, nf, nc, l)

num_coarse_cells = num_cells(get_triangulation(ms_space.UH))
num_fine_cells = num_cells(get_triangulation(ms_space.Uh))
node_coordinates = get_node_coordinates(get_triangulation(ms_space.Uh))

coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells)

patch_coarse_models = get_patch_triangulation(ms_space, num_coarse_cells, CoarseScale())
patch_fine_models = get_patch_triangulation(ms_space, num_coarse_cells, FineScale())
patch_coarse_trian = lazy_map(Triangulation, patch_coarse_models)

patch_interior_dofs = lazy_map(get_interior_indices, patch_fine_models);
patch_boundary_dofs = lazy_map(get_boundary_indices, patch_fine_models);

patch_fine_spaces = lazy_map(build_patch_fine_spaces, patch_fine_models, Gridap.Arrays.Fill(q, num_fine_cells));

elem_global_node_ids, elem_local_node_ids = get_elem_data(ms_space);
elem_local_unique_node_ids = lazy_map(Broadcasting(get_unique_node_ids), elem_local_node_ids);

A(x) = 1.0;
patch_stima = lazy_map(assemble_stima, patch_fine_spaces, Gridap.Arrays.Fill(A, num_coarse_cells), Gridap.Arrays.Fill(4, num_coarse_cells), patch_interior_dofs);
patch_lmat = lazy_map(assemble_rect_matrix, patch_coarse_trian, patch_fine_spaces, Gridap.Arrays.Fill(p, num_coarse_cells), patch_interior_dofs, elem_local_unique_node_ids);
patch_system = lazy_map(saddle_point_system,patch_stima, patch_lmat);