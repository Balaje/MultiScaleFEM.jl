###### ######## ######## ######## ######## ######## # 
# Program to test the multiscale basis computation  #
###### ######## ######## ######## ######## ######## # 
include("2d_HigherOrderMS.jl")

domain = (0.0, 1.0, 0.0, 1.0)

# Fine scale space description
nf = 2^8
q = 1
nc = 2^1
p = 1
l = 3 # Patch size parameter
ms_space = MultiScaleFESpace(domain, q, p, nf, nc, l);

num_coarse_cells = num_cells(ms_space.Ωc);
num_fine_cells = num_cells(get_triangulation(ms_space.Uh));
node_coordinates = get_node_coordinates(get_triangulation(ms_space.Uh));

coarse_to_fine_elems = get_coarse_to_fine_map(num_coarse_cells, num_fine_cells);

patch_coarse_models = get_patch_triangulation(ms_space, num_coarse_cells, CoarseScale()) |> collect;
patch_fine_models = get_patch_triangulation(ms_space, num_coarse_cells, FineScale()) |> collect;

patch_local_interior_dofs = lazy_map(get_interior_indices, patch_fine_models) |> collect;
patch_local_boundary_dofs = lazy_map(get_boundary_indices, patch_fine_models) |> collect;

patch_global_to_local, patch_local_to_global = ms_space.patch_global_local_map;
patch_global_interior_dofs = lazy_map(convert_dofs, patch_local_interior_dofs, patch_local_to_global) |> collect;
patch_global_boundary_dofs = lazy_map(convert_dofs, patch_local_boundary_dofs, patch_local_to_global) |> collect;


## Get all the full systems
# D = CellField(rand(num_cells(get_triangulation(ms_space.Uh))), get_triangulation(ms_space.Uh));
D = CellField(1.0, get_triangulation(ms_space.Uh));
elem_global_node_ids = get_elem_data(ms_space);
elem_local_unique_node_ids = lazy_map(get_unique_node_ids, elem_global_node_ids) |> collect;
full_stima = assemble_stima(ms_space.Uh, D, 4);
full_lmat = assemble_rect_matrix(ms_space.Ωc, ms_space.Uh, p, elem_local_unique_node_ids);
full_rhsmat = assemble_rhs_matrix(ms_space.Ωc, p);

patch_coarse_elems = lazy_map(get_patch_coarse_elem, 
  Gridap.Arrays.Fill(ms_space.Ωc, num_coarse_cells),
  Gridap.Arrays.Fill(ms_space.elemTree, num_coarse_cells), 
  Gridap.Arrays.Fill(l, num_coarse_cells), 1:num_coarse_cells) |> collect;

# # Uncomment to check the patch-wise matrices.  
# patch_stima = lazy_map(getindex, Gridap.Arrays.Fill(full_stima, num_coarse_cells), patch_global_interior_dofs, patch_global_interior_dofs);
# patch_lmat = lazy_map(getindex, Gridap.Arrays.Fill(full_lmat, num_coarse_cells), patch_global_interior_dofs, [3p*i-3p+1:3p*i for i in 1:num_coarse_cells]);
# patch_rhs = lazy_map(getindex, Gridap.Arrays.Fill(full_rhsmat, num_coarse_cells), [3p*i-3p+1:3p*i for i in 1:num_coarse_cells], [3p*i-3p+1:3p*i for i in 1:num_coarse_cells]);

# Get the multiscale basis vectors
basis_vec_ms = get_ms_bases(full_stima, full_lmat, full_rhsmat, patch_global_interior_dofs, p);

function visualize_basis_vector(fespace::FESpace, basis_vec_ms, inds)
  for i in inds
    bi = FEFunction(fespace, basis_vec_ms[:,i])
    writevtk(get_triangulation(fespace), "multiscale-bases-"*string(i), cellfields=["Λᵐˢ"=>bi])
  end
end

## Solve the Poisson equation using multiscale method
f(x) = 2π^2*sin(π*x[1])*sin(π*x[2]);
dΩ = Measure(get_triangulation(ms_space.Uh), 4);
lh(v) = ∫(f*v)dΩ;
full_loadvec = assemble_vector(lh, ms_space.Uh);

Kₘₛ = basis_vec_ms'*full_stima*basis_vec_ms;
Fₘₛ = basis_vec_ms'*full_loadvec;
sol = Kₘₛ\Fₘₛ;

full_sol = basis_vec_ms*sol;

uH = FEFunction(ms_space.Uh, full_sol);
writevtk(get_triangulation(ms_space.Uh), "sol_ms", cellfields=["u(x)"=>uH])