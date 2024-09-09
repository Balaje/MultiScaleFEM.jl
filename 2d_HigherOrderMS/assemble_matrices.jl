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
  coarse_cell_coords = lazy_map(Broadcasting(Reindex(get_node_coordinates(coarse_trian))), get_cell_node_ids(coarse_trian))
  num_patch_coarse_cells = num_cells(coarse_trian)
  Ω_fine = get_triangulation(fine_space)
  dΩ_fine = Measure(Ω_fine, p+1)
  n_monomials = (p+1)^2
  L = spzeros(Float64, num_nodes(Ω_fine), num_patch_coarse_cells*n_monomials)    
  index = 1
  coarse_cell_coords = get_cell_coordinates(coarse_trian)
  αβ = poly_exps(p)
  for i=1:num_patch_coarse_cells
    coarse_cell_coord = coarse_cell_coords[i]
    nds_x = (coarse_cell_coord[1][1], coarse_cell_coord[2][1])
    nds_y = (coarse_cell_coord[1][2], coarse_cell_coord[3][2])
    node_ids = patch_local_node_ids[i]   
    for j=1:n_monomials
      b = x->ℳ(x, nds_x, nds_y, p, αβ[j])    
      lh(v) = ∫(b*v)dΩ_fine
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
  coarse_cell_coords = lazy_map(Broadcasting(Reindex(get_node_coordinates(coarse_trian))), get_cell_node_ids(coarse_trian))
  num_patch_coarse_cells = num_cells(coarse_trian)
  Q = CellQuadrature(coarse_trian, 2p)
  cellmaps = get_cell_map(coarse_trian)
  n_monomials = (p+1)^2
  L = spzeros(Float64, num_patch_coarse_cells*n_monomials, num_patch_coarse_cells*n_monomials)  
  index = 1
  coarse_cell_coords = get_cell_coordinates(coarse_trian)  
  αβ = poly_exps(p)
  for i=1:num_patch_coarse_cells
    coarse_cell_coord = coarse_cell_coords[i]
    Qi = get_data(Q)[i]
    cellmap = cellmaps[i]
    Qpoints = lazy_map(Broadcasting(cellmap), Qi.coordinates)
    Qweights = Qi.weights
    nds_x = (coarse_cell_coord[1][1], coarse_cell_coord[2][1])
    nds_y = (coarse_cell_coord[1][2], coarse_cell_coord[3][2])
    for j=1:n_monomials, k=1:n_monomials
      bj = x->ℳ(x, nds_x, nds_y, p, αβ[j])
      bk = x->ℳ(x, nds_x, nds_y, p, αβ[k])
      for q=1:lastindex(Qpoints)
        L[index+j-1, index+k-1] += Qweights[q]*bj(Qpoints[q])*bk(Qpoints[q])
      end
    end
    index += n_monomials
  end
  L
end

"""
Scaled monomial bases at each coarse rectanles. 
The L² functions in the higher order MS Method is the tensor product of the Legendre polynomials in the cell
"""
function ℳ(x::Point, nds_x::NTuple{2,Float64}, nds_y::NTuple{2,Float64}, p::Int64, αβ::NTuple{2,Int64})  
  α,β = αβ
  Λₖ!(x[1], nds_x, p, α+1)*Λₖ!(x[2], nds_y, p, β+1)
end

function LP!(cache::Vector{Float64}, x::Float64)
  p = size(cache,1) - 1
  if(p==0)
    cache[1] = 1.0
  elseif(p==1)
    cache[1] = 1.0
    cache[2] = x      
  else
    cache[1] = 1.0
    cache[2] = x
    for j=2:p
      cache[j+1] = (2j-1)/j*x*cache[j] - (j-1)/j*cache[j-1]
    end
  end
  cache
end  
"""
Shifted Legendre Polynomial with support (a,b)
"""
function Λₖ!(x, nds::Tuple{Float64,Float64}, p::Int64, j::Int64)
  a,b = nds
  cache = Vector{Float64}(undef, p+1)
  fill!(cache,0.0)
  if(a < x < b)
    xhat = -(b+a)/(b-a) + 2.0*x/(b-a)
    LP!(cache, xhat)
  end
  cache[j]
end

function poly_exps(p::Int64)
  X = ones(Int64,p+1)*(0:1:p)';
  Y = (0:1:p)*ones(Int64,p+1)';
  map((a,b)->(a,b), X, Y)
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