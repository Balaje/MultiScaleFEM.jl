###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^3
q = 1
nc = 2^1
p = 1
ms_space = MultiScaleFESpace(domain, q, p, nf, nc)

num_coarse_cells = num_cells(get_triangulation(ms_space.UH))
num_fine_cells = num_cells(get_triangulation(ms_space.Uh))
σ = get_cell_node_ids(get_triangulation(ms_space.Uh))
node_coordinates = get_node_coordinates(get_triangulation(ms_space.Uh))

coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells)

l = 1; # Patch size parameter

# Get the element indices 
patch_fine_elems = get_patch_fine_elems(ms_space, l, num_coarse_cells, coarse_to_fine_elems)

patch_fine_node_ids = lazy_map(Broadcasting(Reindex(σ)), patch_fine_elems)
patch_cell_coordinates = get_patch_cell_coordinates(node_coordinates, patch_fine_node_ids)
patch_node_coordinates = lazy_map(get_patch_node_coordinates, patch_cell_coordinates)
patch_cell_types = get_patch_cell_type(get_cell_type(get_triangulation(ms_space.Uh)), patch_fine_elems)

patch_grids = lazy_map()