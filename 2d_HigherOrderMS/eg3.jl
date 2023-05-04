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
l = 1 # Patch size parameter
ms_space = MultiScaleFESpace(domain, q, p, nf, nc, l);

num_coarse_cells = num_cells(ms_space.Ωc);
num_fine_cells = num_cells(get_triangulation(ms_space.Uh));
node_coordinates = get_node_coordinates(get_triangulation(ms_space.Uh));

coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells);

patch_coarse_models = get_patch_triangulation(ms_space, num_coarse_cells, CoarseScale());
patch_fine_models = get_patch_triangulation(ms_space, num_coarse_cells, FineScale());

patch_local_interior_dofs = lazy_map(get_interior_indices, patch_fine_models);
patch_local_boundary_dofs = lazy_map(get_boundary_indices, patch_fine_models);

patch_global_to_local, patch_local_to_global = ms_space.patch_global_local_map;
patch_global_interior_dofs = lazy_map(convert_dofs, patch_local_interior_dofs, patch_local_to_global);
patch_global_boundary_dofs = lazy_map(convert_dofs, patch_local_boundary_dofs, patch_local_to_global);


## Get all the full systems
A(x) = 1.0;
elem_global_node_ids = get_elem_data(ms_space);
elem_local_unique_node_ids = lazy_map(get_unique_node_ids, elem_global_node_ids);
full_stima = assemble_stima(ms_space.Uh, A, 4);
full_lmat = assemble_rect_matrix(ms_space.Ωc, ms_space.Uh, p, elem_local_unique_node_ids);
full_rhsmat = assemble_rhs_matrix(ms_space.Ωc, p);

patch_coarse_elems = lazy_map(get_patch_coarse_elem, 
  Gridap.Arrays.Fill(ms_space.Ωc, num_coarse_cells),
  Gridap.Arrays.Fill(ms_space.elemTree, num_coarse_cells), 
  Gridap.Arrays.Fill(l, num_coarse_cells), 1:num_coarse_cells);

patch_stima = lazy_map(getindex, Gridap.Arrays.Fill(full_stima, num_coarse_cells), patch_global_interior_dofs, patch_global_interior_dofs);
patch_lmat = lazy_map(getindex, Gridap.Arrays.Fill(full_lmat, num_coarse_cells), patch_global_interior_dofs, [3p*i-3p+1:3p*i for i in 1:num_coarse_cells]);
patch_rhs = lazy_map(getindex, Gridap.Arrays.Fill(full_rhsmat, num_coarse_cells), [3p*i-3p+1:3p*i for i in 1:num_coarse_cells], [3p*i-3p+1:3p*i for i in 1:num_coarse_cells]);
patch_system = lazy_map(saddle_point_system, patch_stima, patch_lmat);


# patch_fine_spaces = lazy_map(build_patch_fine_spaces, patch_fine_models, Gridap.Arrays.Fill(q, num_fine_cells));
# 
# patch_stima = lazy_map(assemble_stima, patch_fine_spaces, Gridap.Arrays.Fill(A, num_coarse_cells), Gridap.Arrays.Fill(4, num_coarse_cells), patch_interior_dofs);
# patch_lmat = lazy_map(assemble_rect_matrix, patch_coarse_trian, patch_fine_spaces, Gridap.Arrays.Fill(p, num_coarse_cells), patch_interior_dofs, elem_local_unique_node_ids);
# patch_system = lazy_map(saddle_point_system,patch_stima, patch_lmat);