include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^9
q = 1
nc = 2^7
p = 1
l = 1 # Patch size parameter

model_h = simplexify(CartesianDiscreteModel(domain, (nf,nf)));
Ωf = Triangulation(model_h);
model_H = simplexify(CartesianDiscreteModel(domain, (nc,nc)));
Ωc = Triangulation(model_H);
σ = get_cell_node_ids(Ωc);
R = vec(map(x->SVector(Tuple(x)), σ));
tree = BruteTree(R, ElemDist());
num_coarse_cells = num_cells(Ωc);
num_fine_cells = num_cells(Ωf);

# nc = 2^4, nf = 2^9, l=1
# Time taken: 1.006 (Approx) [Umeå PC]
coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells);
#@btime get_coarse_to_fine_map($num_coarse_cells, $num_fine_cells);

# nc = 2^4, nf = 2^9, l=1
# Time taken: 5.298 ms (Approx) [Umeå PC]
patch_coarse_elems = lazy_map(get_patch_coarse_elem, 
    lazy_fill(Ωc, num_coarse_cells), 
    lazy_fill(tree, num_coarse_cells), 
    lazy_fill(l, num_coarse_cells), 
    1:num_coarse_cells);
#@btime map($get_patch_coarse_elem, lazy_fill($Ωc, $num_coarse_cells), lazy_fill($tree, $num_coarse_cells), lazy_fill($l, $num_coarse_cells), $(1:num_coarse_cells));

# nc = 2^4, nf = 2^9, l=1
# Time taken: 7.106 ms (Approx) [Umeå PC]
patch_fine_elems = get_patch_fine_elems(patch_coarse_elems, coarse_to_fine_elems);
#@btime get_patch_fine_elems($patch_coarse_elems, $coarse_to_fine_elems);

# Gridap Functions
σ_coarse = get_cell_node_ids(Ωc);
node_coordinates_coarse = get_node_coordinates(Ωc);
cell_types_coarse = get_cell_type(Ωc);
σ_fine = get_cell_node_ids(Ωf);
node_coordinates_fine = get_node_coordinates(Ωf);
cell_types_fine = get_cell_type(Ωf);

# nc = 2^4, nf = 2^9, l=1
# Time taken (greedy): 20.413 s (Approx) [Umeå PC] ---> Bottleneck
# Time taken (lazy): 23.150 ms (Approx) [Umeå PC]
local_to_global_map = get_patch_wise_local_to_global_map(patch_fine_elems, σ_fine);
#@btime get_patch_wise_local_to_global_map($patch_fine_elems, $σ_fine);


# nc = 2^4, nf = 2^9. l=1
# Time taken (Greedy): Lesser than fine-scale
# Time taken (Lazy): 157.042 µs [Umeå PC]
patch_coarse_coords, patch_coarse_local_ids, patch_coarse_cell_types = _patch_model_data(σ_coarse, node_coordinates_coarse, cell_types_coarse, patch_coarse_elems, num_coarse_cells);
#@btime _patch_model_data($σ_coarse, $node_coordinates_coarse, $cell_types_coarse, $patch_coarse_elems, $num_coarse_cells);

patch_grids = map(UnstructuredGrid, patch_coarse_coords, patch_coarse_local_ids, lazy_fill(get_reffes(Ωc), num_coarse_cells), patch_coarse_cell_types);
grids_topology = map(GridTopology, patch_grids);
face_labelling = map(FaceLabeling, grids_topology);
patch_models_coarse = map(DiscreteModel, patch_grids, grids_topology, face_labelling);

# nc = 2^4, nf = 2^9. l=1
# Time taken (Greedy): Takes a long time
# Time taken (Lazy): 71.326 ms [Umeå PC]
patch_fine_coords, patch_fine_local_ids, patch_fine_cell_types = _patch_model_data(σ_fine, node_coordinates_fine, cell_types_fine, patch_fine_elems, num_coarse_cells);
#@btime _patch_model_data($σ_fine, $node_coordinates_fine, $cell_types_fine, $patch_fine_elems, $num_coarse_cells);

# nc = 2^4, nf = 2^9, l=1
patch_grids = lazy_map(UnstructuredGrid, patch_fine_coords, patch_fine_local_ids, lazy_fill(get_reffes(Ωf), num_coarse_cells), patch_fine_cell_types);
grids_topology = lazy_map(GridTopology, patch_grids);
face_labelling = lazy_map(FaceLabeling, grids_topology);
patch_models_fine = lazy_map(DiscreteModel, patch_grids, grids_topology, face_labelling);

# nc = 2^4, nf = 2^9. l=1
# Time taken (Greedy): Takes a long time
# Time taken (Lazy): 90 ms (Approx) [Umeå PC]
patch_local_interior_dofs = lazy_map(get_local_indices, patch_models_fine, lazy_fill("interior", num_coarse_cells));
#@btime lazy_map(get_local_indices, $patch_models_fine, lazy_fill("interior", $num_coarse_cells));

patch_local_boundary_dofs = lazy_map(get_local_indices, patch_models_fine, lazy_fill("boundary", num_coarse_cells));
patch_global_interior_dofs = lazy_map(convert_dofs, patch_local_interior_dofs, local_to_global_map);
patch_global_boundary_dofs = lazy_map(convert_dofs, patch_local_boundary_dofs, local_to_global_map);
interior_boundary_global = (patch_global_interior_dofs, patch_global_boundary_dofs);
interior_boundary_local = (patch_local_interior_dofs, patch_local_boundary_dofs);
