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

# coarse_to_fine_map for 1 level refinement
let 
  function source_to_target(A, B)
    C = Matrix{Int64}(undef, size(A,1), size(A,2)*size(B,2))
    fill!(C,0)
    for i=1:lastindex(A,1), j=1:lastindex(A,2)
      for k=1:lastindex(B,2)
        C[i,k+(j-1)*size(B,2)] = B[A[i,j],k]     
      end
    end 
    C
  end
  function refine_once(j::Int64)    
    check_elem(k, j) = (k%2^(j+1) == 1)
    num_cells = 2^(2*j+1)
    ref_1 = Matrix{Int64}(undef, num_cells, 4)
    ref_1[1,:] = [1,2,3,2^(j+2)+1]
    for k=2:num_cells
      tmp = sort(ref_1[k-1,:], rev=true)
      kk = ceil(Int64, k/(2^(j+1)))
      if((k % 2) == 0)
        ref_1[k,:] = [tmp[2]+1; tmp[1]+1:tmp[1]+3] .+ check_elem(k,j)*kk^0*2^(j+2)
      else
        ref_1[k,:] = [tmp[end]+1:tmp[end]+3; tmp[1]+1] .+ check_elem(k,j)*kk^0*2^(j+2)
      end
    end
    ref_1
  end
  function coarse_to_fine_map(num_coarse_cells::Int64, num_fine_cells::Int64)        
    j_coarse = ceil(Int64, 0.5*(log2(num_coarse_cells)-1))    
    j_fine = ceil(Int64, 0.5*(log2(num_fine_cells)-1))
    @show j_coarse, j_fine
    all_j = j_coarse+1:j_fine
    c_to_f_maps = [refine_once(j-1) for j in all_j]
    c_to_f = source_to_target(c_to_f_maps[end-1], c_to_f_maps[end])
    for l=lastindex(all_j)-1:-1:2
      c_to_f = source_to_target(c_to_f_maps[l-1], c_to_f)
    end
    c_to_f
  end  
  global coarse_to_fine_maps = coarse_to_fine_map(2^1, 2^7)  
end