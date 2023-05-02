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

function assemble_stima(fine_space::FESpace, A::Function, qorder::Int64, interior_dofs)
  Ω = get_triangulation(fine_space)
  dΩ = Measure(Ω, qorder)
  a(u,v) = ∫( A*∇(u) ⊙ ∇(v) )dΩ
  stima = assemble_matrix(a, fine_space, fine_space)  
  stima[interior_dofs, interior_dofs]
end

function assemble_rect_matrix(coarse_trian::Triangulation, fine_space::FESpace, p::Int64, interior_dofs, patch_local_node_ids)
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
      b = x->coarse_scale_bases(x, centroid, diameter, αβ[j])    
      lh(v) = ∫(b*v)dΩ
      L[patch_local_node_ids[i],index] += assemble_vector(lh, fine_space)[patch_local_node_ids[i]]
      index = index+1
    end
  end
  L[interior_dofs, :]
end

### Scaled monomial bases at each coarse triangles
function coarse_scale_bases(x::Point, centroid, diameter, poly_exps)    
  ℳ(x, centroid, diameter, poly_exps)
end

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