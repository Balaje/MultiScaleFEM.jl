##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #
# Contains the interfaces to build the coarse scale fespaces and the saddle point matrices
##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #

"""
Function to assemble the standard stiffness matrix given the diffusion coefficient A(x)
"""
function assemble_stima(fespace::FESpace, A, qorder::Int64)
  Ω = get_triangulation(fespace)
  dΩ = Measure(Ω, qorder)
  a(u,v) = ∫( A*∇(u) ⊙ ∇(v) )dΩ
  assemble_matrix(a, fespace, fespace)  
end

"""
Function to assemble the standard mass matrix given the reaction coefficient A(x)
"""
function assemble_massma(fespace::FESpace, A, qorder::Int64)
  Ω = get_triangulation(fespace)
  dΩ = Measure(Ω, qorder)
  a(u,v) = ∫( A*(u)⋅(v) )dΩ
  assemble_matrix(a, fespace, fespace)  
end


"""
Function to assemble the standard load vector given the load f(x)
"""
function assemble_loadvec(fespace::FESpace, f, qorder::Int64)
  Ω = get_triangulation(fespace)
  dΩ = Measure(Ω, qorder)
  l(v) = ∫( f*v )dΩ
  assemble_vector(l, fespace)
end

"""
Function to assemble the full rectangular matrix in the saddle point system
"""
function assemble_rect_matrix(coarse_trian::Triangulation, fine_space::FESpace, p::Int64, patch_local_node_ids)
  num_patch_coarse_cells = num_cells(coarse_trian)
  Ω = get_triangulation(fine_space)
  dΩ = Measure(Ω, p+1)
  n_monomials = Int64((p+1)*(p+2)*0.5)
  L = spzeros(Float64, num_nodes(Ω), num_patch_coarse_cells*n_monomials)    
  index = 1
  coarse_cell_coords = get_cell_coordinates(coarse_trian)
  centroids = lazy_map(x->sum(x)/3, coarse_cell_coords)
  diameters = lazy_map(x->max(norm(x[1]-x[2]), norm(x[2]-x[3]), norm(x[3]-x[1])), coarse_cell_coords)
  αβ = poly_exps(p)
  for i=1:num_patch_coarse_cells
    node_ids = patch_local_node_ids[i]
    centroid = centroids[i]   
    diameter = diameters[i]    
    for j=1:n_monomials
      b = x->ℳ(x, centroid, diameter, αβ[j])    
      lh(v) = ∫(b*v)dΩ
      L[node_ids,index] += assemble_vector(lh, fine_space)[node_ids]
      index = index+1
    end
  end
  L
end

"""
Function to assemble the inner-product of the L² functions.
"""
function assemble_rhs_matrix(coarse_trian::Triangulation, p::Int64)
  num_patch_coarse_cells = num_cells(coarse_trian)
  Q = CellQuadrature(coarse_trian, 2p)
  cellmaps = get_cell_map(coarse_trian)
  n_monomials = Int64((p+1)*(p+2)*0.5)
  L = spzeros(Float64, num_patch_coarse_cells*n_monomials, num_patch_coarse_cells*n_monomials)  
  index = 1
  coarse_cell_coords = get_cell_coordinates(coarse_trian)
  centroids = lazy_map(x->sum(x)/3, coarse_cell_coords)
  diameters = lazy_map(x->max(norm(x[1]-x[2]), norm(x[2]-x[3]), norm(x[3]-x[1])), coarse_cell_coords)
  αβ = poly_exps(p)
  for i=1:num_patch_coarse_cells
    Qi = get_data(Q)[i]
    cellmap = cellmaps[i]
    Qpoints = lazy_map(Broadcasting(cellmap), Qi.coordinates)
    Qweights = Qi.weights
    centroid = centroids[i]   
    diameter = diameters[i]  
    for j=1:n_monomials, k=1:n_monomials
      bj = x->ℳ(x, centroid, diameter, αβ[j])
      bk = x->ℳ(x, centroid, diameter, αβ[k])
      for q=1:lastindex(Qpoints)
        L[index+j-1, index+k-1] += Qweights[q]*bj(Qpoints[q])*bk(Qpoints[q])
      end
    end
    index += n_monomials
  end
  L
end

"""
Scaled monomial bases at each coarse triangles. 
Acts as the L² functions in the higher order MS Method
"""
function ℳ(x::Point, centroid::Point, dia::Float64, αβ::Tuple{Int64,Int64})
  ((x[1] - centroid[1])/dia)^αβ[1]*((x[2] - centroid[2])/dia)^αβ[2]
end

function poly_exps(p::Int64)
  exps = Vector{Vector{Tuple{Int64,Int64}}}(undef,p+1)
  for i=1:p+1
    exps[i] = Vector{Tuple{Int64,Int64}}(undef, i)    
    for (j,k) in zip((i-1:-1:0), (0:1:i-1))
      exps[i][k+1] = (j,k)      
    end
  end
  reduce(vcat, exps)
end

"""
Function to get the saddle point system given the stiffness and the rectangular matrix
"""
saddle_point_system(stima, lmat) = [stima lmat; lmat' spzeros(size(lmat,2), size(lmat,2))]


function assemble_ms_matrix!(global_matrix, elem_matrices, p, KorM)
  fill!(global_matrix,0.0)
  num_coarse_cells = size(elem_matrices,1)
  n_monomials = Int((p+1)*(p+2)*0.5)
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  patch_coarse_dof = lazy_map(Broadcasting(elem_to_dof), 1:num_coarse_cells)  
  B = collect(elem_matrices)
  for i=1:num_coarse_cells, j=1:num_coarse_cells
    global_matrix[patch_coarse_dof[i], patch_coarse_dof[j]] += B[i]'*KorM*B[j]
  end
  global_matrix
end
function assemble_ms_load!(global_load, elem_vectors, p, F)
  fill!(global_load,0.0);
  num_coarse_cells = size(elem_vectors,1);
  n_monomials = Int((p+1)*(p+2)*0.5)
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  patch_coarse_dof = lazy_map(Broadcasting(elem_to_dof), 1:num_coarse_cells)  
  B = collect(elem_vectors)
  for i=1:num_coarse_cells
    global_load[patch_coarse_dof[i]] += B[i]'*F    
  end
  global_load
end

function assemble_ms_matrix(elem_matrices, p, KorM)
  num_coarse_cells = size(elem_matrices, 1)
  n_monomials = Int((p+1)*(p+2)*0.5)
  global_matrix = zeros(Float64, num_coarse_cells*n_monomials, num_coarse_cells*n_monomials)
  assemble_ms_matrix!(global_matrix, elem_matrices, p, KorM)
end

function assemble_ms_load(elem_vecs, p, F)
  num_coarse_cells = size(elem_vecs, 1)
  n_monomials = Int((p+1)*(p+2)*0.5)
  global_vec = zeros(Float64, num_coarse_cells*n_monomials)  
  assemble_ms_load!(global_vec, elem_vecs, p, F)
end

function get_fine_scale!(res, elem_vecs, p, solms)
  fill!(res,0.0)
  num_coarse_cells = size(elem_vecs, 1)
  n_monomials = Int((p+1)*(p+2)*0.5)
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  patch_coarse_dof = lazy_map(Broadcasting(elem_to_dof), 1:num_coarse_cells)  
  for i=1:num_coarse_cells
    res .+= elem_vecs[i]*solms[patch_coarse_dof[i]]
  end
  res
end

function get_fine_scale(elem_vecs, p, solms)
  res = zeros(Float64, size(elem_vecs[1],1))
  get_fine_scale!(res, elem_vecs, p, solms)
  res
end