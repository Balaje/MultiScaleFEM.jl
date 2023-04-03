###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)
nf = 2^1
q = 1
fine_scale_space = FineScaleSpace(domain, q, nf)

l = 1
el = 2
patch_node_coordinates = collect(get_patch_node_coordinates(fine_scale_space, l, el));
patch_cell_ids = Table(get_patch_local_node_ids(fine_scale_space, l, el));
patch_cell_type = collect(get_patch_cell_types(fine_scale_space, l, el));
patch_grid = UnstructuredGrid(patch_node_coordinates, patch_cell_ids, get_reffes(get_triangulation(fine_scale_space.U)), patch_cell_type);
patch_model = DiscreteModel(patch_grid, GridTopology(patch_grid), FaceLabeling(GridTopology(patch_grid)));
reffe = ReferenceFE(lagrangian, Float64, q)
U = TestFESpace(patch_model, reffe, conformity=:H1, dirichlet_tags="boundary")