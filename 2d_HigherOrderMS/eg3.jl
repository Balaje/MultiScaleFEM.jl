###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^5
q = 1
fine_scale_space = FineScaleSpace(domain, q, nf)
num_cells_fine = num_cells(get_triangulation(fine_scale_space.U))

# Coarse scale space description
nc = 2^1
p = 1
coarse_scale_space = FineScaleSpace(domain, p, nc)
num_cells_coarse = num_cells(get_triangulation(coarse_scale_space.U))

aspect_ratio = Int64(sqrt(num_cells_fine/num_cells_coarse))
el = 1
l = 1
patch_node_coordinates = collect(get_patch_node_coordinates(fine_scale_space, l, el));
patch_cell_ids = Table(get_patch_local_node_ids(fine_scale_space, l, el));
patch_cell_type = collect(get_patch_cell_types(fine_scale_space, l, el));
patch_grid = UnstructuredGrid(patch_node_coordinates, patch_cell_ids, get_reffes(get_triangulation(fine_scale_space.U)), patch_cell_type);
patch_model = DiscreteModel(patch_grid, GridTopology(patch_grid), FaceLabeling(GridTopology(patch_grid)));
reffe = ReferenceFE(lagrangian, Float64, q)
U = TestFESpace(patch_model, reffe, conformity=:H1, dirichlet_tags="boundary")

# include("eg3.jl");  writevtk(patch_model, "model") # Run this and visualize in Paraview