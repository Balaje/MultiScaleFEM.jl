include("2d_HigherOrderMS.jl")

using BenchmarkTools

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^9
q = 1
nc = 2^4
p = 1
l = 3 # Patch size parameter

model_h = simplexify(CartesianDiscreteModel(domain, (nf,nf)));
Ωf = Triangulation(model_h);
σ_fine = get_cell_node_ids(Ωf);
model_H = simplexify(CartesianDiscreteModel(domain, (nc,nc)));
Ωc = Triangulation(model_H);
σ_coarse = get_cell_node_ids(Ωc);
R = vec(map(x->SVector(Tuple(x)), σ_coarse));
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

patch_fine_node_ids = get_patch_global_node_ids(patch_fine_elems, σ_fine);
#@btime get_patch_global_node_ids($patch_fine_elems, $σ_fine);

interior_global = lazy_map(get_interior_indices_direct, patch_fine_node_ids);
boundary_global = lazy_map(get_boundary_indices_direct, patch_fine_node_ids);
#@btime map(get_interior_indices_direct, $patch_fine_node_ids);
#@btime map(get_boundary_indices_direct, $patch_fine_node_ids);
interior_boundary_global = (interior_global, boundary_global);
