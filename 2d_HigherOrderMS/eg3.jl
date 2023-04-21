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
ms_space = MultiScaleFESpace(domain, q, p, nf, nc)

num_coarse_cells = num_cells(get_triangulation(ms_space.UH))
num_fine_cells = num_cells(get_triangulation(ms_space.Uh))
node_coordinates = get_node_coordinates(get_triangulation(ms_space.Uh))

coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells)

l = 1; # Patch size parameter

patch_coarse_models = get_patch_triangulation(ms_space, l, num_coarse_cells)
patch_fine_models = get_patch_triangulation(ms_space, l, num_coarse_cells, coarse_to_fine_elems)

p = 3; q = 1;
patch_coarse_spaces = lazy_map(build_patch_coarse_spaces, patch_coarse_models, Gridap.Arrays.Fill(p, num_coarse_cells));
patch_fine_spaces = lazy_map(build_patch_fine_spaces, patch_fine_models, Gridap.Arrays.Fill(q, num_fine_cells));

A(x) = 1.0
patch_stima = lazy_map(assemble_stima, patch_fine_spaces, Gridap.Arrays.Fill(A, num_coarse_cells), Gridap.Arrays.Fill(4, num_coarse_cells));