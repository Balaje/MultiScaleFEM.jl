##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #
# Contains the interfaces to build the coarse scale fespaces and the saddle point matrices
##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #

function build_patch_fine_spaces(model::DiscreteModel, q::Int64)
  ref_space = ReferenceFE(lagrangian, Float64, q)
  FESpace(model, ref_space, conformity=:H1)
end

function get_boundary_indices(model::DiscreteModel)
  fl = get_face_labeling(model)
  findnz(sparse(get_face_tag(fl, "boundary", 0)))[1]
end

function get_interior_indices(model::DiscreteModel)
  fl = get_face_labeling(model)
  findnz(sparse(get_face_tag(fl, "interior", 0)))[1]
end

function get_full_indices(model::DiscreteModel)
  1:num_dofs()
end

function assemble_stima(fine_space::FESpace, A, qorder::Int64)
  Ω = get_triangulation(fine_space)
  dΩ = Measure(Ω, qorder)
  a(u,v) = ∫( A*∇(u) ⊙ ∇(v) )dΩ
  assemble_matrix(a, fine_space, fine_space)  
end

function assemble_rect_matrix(coarse_trian::Triangulation, fine_space::FESpace, p::Int64, patch_local_node_ids)
  num_patch_coarse_cells = num_cells(coarse_trian)
  Ω = get_triangulation(fine_space)
  dΩ = Measure(Ω, p+1)
  L = spzeros(Float64, num_nodes(Ω), num_patch_coarse_cells*3*p)  
  n_monomials = Int64((p+1)*(p+2)*0.5)
  index = 1
  coarse_cell_coords = get_cell_coordinates(coarse_trian)
  centroids = lazy_map(x->sum(x)/3, coarse_cell_coords)
  diameters = lazy_map(x->max(norm(x[1]-x[2]), norm(x[2]-x[3]), norm(x[3]-x[1])), coarse_cell_coords)
  αβ = poly_exps(p)
  for i=1:num_patch_coarse_cells
    centroid = centroids[i]   
    diameter = diameters[i]    
    for j=1:n_monomials
      b = x->ℳ(x, centroid, diameter, αβ[j])    
      lh(v) = ∫(b*v)dΩ
      L[patch_local_node_ids[i],index] += assemble_vector(lh, fine_space)[patch_local_node_ids[i]]
      index = index+1
    end
  end
  L
end

function assemble_rhs_matrix(coarse_trian::Triangulation, p::Int64)
  num_patch_coarse_cells = num_cells(coarse_trian)
  Q = CellQuadrature(coarse_trian, 2p)
  cellmaps = get_cell_map(coarse_trian)
  L = spzeros(Float64, num_patch_coarse_cells*3p, num_patch_coarse_cells*3p)
  n_monomials = Int64((p+1)*(p+2)*0.5)
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
    index += 3p
  end
  L
end

### Scaled monomial bases at each coarse triangles
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

saddle_point_system(stima, lmat) = [stima lmat; lmat' spzeros(size(lmat,2), size(lmat,2))]

function get_ms_bases(stima::SparseMatrixCSC{Float64, Int64}, lmat::SparseMatrixCSC{Float64,Int64}, rhsmat::SparseMatrixCSC{Float64,Int64}, interior_dofs, p::Int64)  
  n_fine_dofs = size(lmat, 1)
  n_coarse_dofs = size(lmat, 2)
  num_coarse_cells = size(interior_dofs, 1)
  coarse_dofs = [3p*i-3p+1:3p*i for i in 1:num_coarse_cells]
  basis_vec_ms = spzeros(Float64, n_fine_dofs, n_coarse_dofs)
  patch_stima = lazy_map(getindex, Gridap.Arrays.Fill(stima, num_coarse_cells), interior_dofs, interior_dofs);
  patch_lmat = lazy_map(getindex, Gridap.Arrays.Fill(lmat, num_coarse_cells), interior_dofs, coarse_dofs);
  patch_rhs = lazy_map(getindex, Gridap.Arrays.Fill(rhsmat, num_coarse_cells), coarse_dofs, coarse_dofs);
  for i=1:num_coarse_cells
    LHS = saddle_point_system(patch_stima[i], patch_lmat[i])
    RHS = [zeros(Float64, size(interior_dofs[i],1), size(coarse_dofs[i],1)); collect(patch_rhs[i])]
    sol = LHS\RHS
    for j=1:lastindex(coarse_dofs[i])
      basis_vec_ms[interior_dofs[i], coarse_dofs[i][j]] = sol[1:length(interior_dofs[i]), j]
    end
  end
  basis_vec_ms
end