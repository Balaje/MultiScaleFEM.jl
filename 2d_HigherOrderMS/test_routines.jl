include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^4
q = 1
nc = 2^2
p = 1
l = 1 # Patch size parameter
ms_space = MultiScaleFESpace(domain, q, p, nf, nc, l);

num_coarse_cells = num_cells(get_triangulation(ms_space.UH));
num_fine_cells = num_cells(get_triangulation(ms_space.Uh));

coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells);

patch_coarse_elems = lazy_map(get_patch_coarse_elem, 
  Gridap.Arrays.Fill(ms_space.UH, num_coarse_cells),
  Gridap.Arrays.Fill(ms_space.elemTree, num_coarse_cells), 
  Gridap.Arrays.Fill(l, num_coarse_cells), 1:num_coarse_cells);

patch_fine_elems = get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elems);

σ_fine = get_cell_node_ids(get_triangulation(ms_space.Uh));

patch_fine_global_node_ids = get_patch_global_node_ids(patch_fine_elems, σ_fine);

unique_patch_fine_node_ids = lazy_map(get_unique_node_ids, patch_fine_global_node_ids);

global_to_local = lazy_map(get_local_node_ids, unique_patch_fine_node_ids);

coarse_elems_node_ids = lazy_map(Broadcasting(Reindex(σ_fine)), patch_fine_elems);

patch_local_node_ids = lazy_map(convert_global_to_local_node_ids, coarse_elems_node_ids, global_to_local);