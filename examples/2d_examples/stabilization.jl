using LazyArrays
using MultiscaleFEM.MultiscaleBases

domain = (0.0, 1.0, 0.0, 1.0);


nf = 2^7;
nc = 2^2;
p = 1;
l = 5; # Patch size parameter

# Background fine scale discretization
FineScale = FineTriangulation(domain, nf);
reffe = ReferenceFE(lagrangian, Float64, 1);

# Coarse scale discretization
CoarseScale = CoarseTriangulation(domain, nc, l);

# Multiscale Triangulation
Ωₘₛ = MultiScaleTriangulation(CoarseScale, FineScale);

Ωc = Ωₘₛ.Ωc
num_coarse_cells = nc^2
patch_coarse_elems = BroadcastVector(MultiscaleBases.get_patch_coarse_elem, 
                    lazy_fill(Ωc.trian, num_coarse_cells), 
                    lazy_fill(Ωc.tree, num_coarse_cells), 
                    lazy_fill(1, num_coarse_cells), 
                    1:num_coarse_cells);

σ_coarse = Gridap.Geometry.get_cell_node_ids(Ωc.trian);
patch_node_ids = lazy_map(Broadcasting(Reindex(σ_coarse)), patch_coarse_elems);
function dist_fun(x::AbstractVector, y)
  dist = abs(x[1] - y)
  for i=1:lastindex(x), j=1:lastindex(y)
    dist = min(dist, abs(x[i]-y))
  end
  dist
end

function find_elements_in_patch(Ωc::CoarseTriangulation, el::Integer, domain) 
  nc = num_cells(Ωc.trian) 
  num_coarse_cells = nc^2
  patch_coarse_elems = BroadcastVector(MultiscaleBases.get_patch_coarse_elem, 
                                    lazy_fill(Ωc.trian, num_coarse_cells), 
                                    lazy_fill(Ωc.tree, num_coarse_cells), 
                                    lazy_fill(1, num_coarse_cells), 
                                    1:num_coarse_cells);
  σ_coarse = Gridap.Geometry.get_cell_node_ids(Ωc.trian);
  cell_coords = Gridap.Geometry.get_cell_coordinates(Ωc.trian);
  patch_node_ids = lazy_map(Broadcasting(Reindex(σ_coarse)), patch_coarse_elems);
  if(length(patch_coarse_elems[el]) == 9)
    el_ids = zeros(Int64, 4, 4)
    el_ids_local = zeros(Int64, 4, 4)
    for i=1:4 # Loop through local ids
      node_ids = patch_node_ids[el]
      el_node_ids = node_ids[5]      
      distances = dist_fun.(node_ids, Ref(el_node_ids[i]))      
      # el_ids[:,i] = setdiff(patch_coarse_elems[el][findall(distances .≈ 0)], el)
      # el_ids_local[:,i] = sort(setdiff(1:4, i), rev=true)
      el_ids[:,i] = patch_coarse_elems[el][findall(distances .≈ 0)]      
      el_ids_local[:,i] = sort(1:4, rev=true)
    end    
    return el_ids, el_ids_local    
  elseif(length(patch_coarse_elems[el]) == 6)
    # Get the position of the element
    cell_coord = cell_coords[el]    
    x_coord_1 = cell_coord[1][1]; x_coord_4 = cell_coord[4][1]
    y_coord_2 = cell_coord[2][2]; y_coord_3 = cell_coord[3][2]
    if(x_coord_1 == domain[1])
      local_inds = [2,4] # left boundary
      local_el = 3 # local position of element
    elseif(x_coord_4 == domain[2])
      local_inds = [1,3] # right boundary
      local_el = 4 # local position of element
    elseif(y_coord_2 == domain[3])
      local_inds = [3,4] # bottom boundary
      local_el = 2 # local position of element
    elseif(y_coord_3 == domain[4])
      local_inds = [1,2] # top boundary
      local_el = 5 # local position of element
    end
    # The variable local_inds returns the local indices of the interior node indices
    el_ids = zeros(Int64, 4, 2) # To store the global element indices
    el_ids_local = zeros(Int64, 4, 2) # To store the local node indices
    node_ids = patch_node_ids[el]
    el_node_ids = node_ids[local_el][local_inds]                   
    for (i,j)=zip(1:4,local_inds)      
      distances = dist_fun.(node_ids, Ref(el_node_ids[i]))            
      # el_ids[:,i] = setdiff(patch_coarse_elems[el][findall(distances .≈ 0)], el)
      # el_ids_local[:,i] = sort(setdiff(1:4, j), rev=true)
      el_ids[:,i] = patch_coarse_elems[el][findall(distances .≈ 0)]      
      el_ids_local[:,i] = sort(1:4, rev=true)
    end
    return el_ids, el_ids_local
  elseif(length(patch_coarse_elems[el]) == 4)
    # Get the position of the element
    cell_coord = cell_coords[el]  
    x_coord_1, y_coord_1 = cell_coord[1]; x_coord_4, y_coord_4 = cell_coord[4]
    x_coord_2, y_coord_2 = cell_coord[2]; x_coord_3, y_coord_3 = cell_coord[3]

    if((x_coord_1,y_coord_1)==(domain[1],domain[3]))
      local_inds = 4
      local_el = 1
    elseif((x_coord_2,y_coord_2)==(domain[2],domain[3]))
      local_inds = 3
      local_el = 2
    elseif((x_coord_3,y_coord_3)==(domain[1],domain[4]))
      local_inds = 2
      local_el = 3
    elseif((x_coord_4,y_coord_4)==(domain[2],domain[4]))
      local_inds = 1
      local_el = 4
    end

    node_ids = patch_node_ids[el]
    el_node_ids = node_ids[local_el][local_inds]
    distances = dist_fun.(node_ids, Ref(el_node_ids))
    # el_ids = setdiff(patch_coarse_elems[el][findall(distances .≈ 0)], el)
    # el_ids_local = sort(setdiff(1:4, local_inds), rev=true)    
    el_ids = patch_coarse_elems[el][findall(distances .≈ 0)]      
    el_ids_local = sort(1:4, rev=true)
    return reshape(el_ids,:,1), reshape(el_ids_local,:,1)
  end
end

function ϕᵣ(x)    
  if((-1 < x[1] <= 1) && (-1 < x[2] <= 1) )
    return (1/4*(1-x[1])*(1-x[2]), 1/4*(1+x[1])*(1-x[2]), 1/4*(1-x[1])*(1+x[2]), 1/4*(1+x[1])*(1+x[2]))
  else
    return (0.0, 0.0, 0.0, 0.0)
  end
end

function χ(x, cell_coord)  
  x₁,x₂ = cell_coord[1][1], cell_coord[4][1]
  y₁,y₂ = cell_coord[2][2], cell_coord[3][2]
  x̂ = -(x₁+x₂)/(x₂-x₁) + 2.0*x[1]/(x₂-x₁)
  ŷ = -(y₁+y₂)/(y₂-y₁) + 2.0*x[2]/(y₂-y₁)
  (x̂, ŷ)
end

function ϕ(x, Ωc::CoarseTriangulation, el, domain, lel)
  cell_coords = Gridap.Geometry.get_cell_coordinates(Ωc.trian);
  global_elem_ids, local_node_ids = find_elements_in_patch(Ωc, el, domain)
  M, N = size(global_elem_ids)
  res = zeros(Float64, M, N)
  # res = 0.0
  for j=1:N
    for i=1:M
      cell_coord = cell_coords[global_elem_ids[i,j]]     
      res[i,j] += (ϕᵣ(χ(x, cell_coord)))[local_node_ids[i,j]]         
    end
  end
  sum(res[:,lel])
end