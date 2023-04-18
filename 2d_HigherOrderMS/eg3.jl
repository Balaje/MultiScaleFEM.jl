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


function save_models(patch_coarse_models, patch_fine_models, i)
  writevtk(patch_coarse_models[i], "2d_HigherOrderMS\\"*string(i)*"-th_coarse_model")
  writevtk(patch_fine_models[i], "2d_HigherOrderMS\\"*string(i)*"-th_fine_model")
end