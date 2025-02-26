module Stabilization

using Gridap
using SparseArrays
using StaticArrays
using SplitApplyCombine
using LazyArrays
using MPI
using LinearAlgebra

using MultiscaleFEM.MultiscaleBases: get_basis_functions
using MultiscaleFEM.MultiscaleBases: CoarseTriangulation, FineTriangulation, MultiScaleTriangulation, MultiScaleFESpace, lazy_fill
using MultiscaleFEM.MultiscaleBases: get_coarse_scale_patch_fine_scale_interior_node_indices, get_coarse_scale_patch_fine_scale_boundary_node_indices
using MultiscaleFEM.MultiscaleBases: get_coarse_scale_patch_coarse_elem_ids, get_coarse_scale_elem_fine_scale_node_indices
using MultiscaleFEM.MultiscaleBases: assemble_rect_matrix, assemble_lm_l2_matrix
using MultiscaleFEM.MultiscaleBases: MultiScaleCorrections, get_basis_functions, build_basis_functions!
using MultiscaleFEM.MultiscaleBases: get_patch_coarse_elem

using MultiscaleFEM.CoarseToFine: coarsen, get_fine_nodes_in_coarse_elems

using MPI
# comm = MPI.COMM_WORLD
# MPI.Init()
# mpi_size = MPI.Comm_size(comm)
# mpi_rank = MPI.Comm_rank(comm)


"""
A distance metric to check whether the elements are connected.
"""
function dist_fun(x::AbstractVector, y)
  dist = abs(x[1] - y)
  for i=1:lastindex(x)
    dist = min(dist, abs(x[i]-y))
  end
  dist
end

"""
Let K be the element we are interested in. 
Function to return the element indices in the patch of K contributing to the basis functions on K
(Could be simplified)
"""
function find_elements_in_patch(Ωc::CoarseTriangulation, el::Integer, domain) 
  num_coarse_cells = num_cells(Ωc.trian)
  patch_coarse_elems = BroadcastVector(get_patch_coarse_elem, 
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
    el_ids = patch_coarse_elems[el][findall(distances .≈ 0)]      
    el_ids_local = sort(1:4, rev=true)
    return reshape(el_ids,:,1), reshape(el_ids_local,:,1)
  end
end

"""
The 2D basis function on the reference domain
"""
function ϕᵣ(x)    
  if((-1 <= x[1] <= 1) && (-1 <= x[2] <= 1) )
    return @SVector[1/4*(1-x[1])*(1-x[2]), 1/4*(1+x[1])*(1-x[2]), 1/4*(1-x[1])*(1+x[2]), 1/4*(1+x[1])*(1+x[2])]
  else
    return @SVector[0.0, 0.0, 0.0, 0.0]
  end
end

function ϕᵣ¹(x)    
  if((-1 < x[1] <= 1) && (-1 <= x[2] < 1) )
    return @SVector[1/4*(1-x[1])*(1-x[2]), 1/4*(1+x[1])*(1-x[2]), 1/4*(1-x[1])*(1+x[2]), 1/4*(1+x[1])*(1+x[2])]
  else
    return @SVector[0.0, 0.0, 0.0, 0.0]
  end
end

"""
Transform the physical coordinate to the reference domain
"""
function χ(x, cell_coord)  
  x₁,x₂ = cell_coord[1][1], cell_coord[4][1]
  y₁,y₂ = cell_coord[2][2], cell_coord[3][2]
  x̂ = -(x₁+x₂)/(x₂-x₁) + 2.0*x[1]/(x₂-x₁)
  ŷ = -(y₁+y₂)/(y₂-y₁) + 2.0*x[2]/(y₂-y₁)
  (x̂, ŷ)
end

"""
The ιₖ(x) function for an element k
"""
function ιₖ(x, Ωc::CoarseTriangulation, global_local_ids)
  cell_coords = Gridap.Geometry.get_cell_coordinates(Ωc.trian);  
  global_elem_ids, local_node_ids =  global_local_ids
  M, N = size(global_elem_ids)
  res = zeros(Float64, M, N)  
  for j=1:N, i=1:M
    cell_coord = cell_coords[global_elem_ids[i,j]]     
    res[i,j] += (ϕᵣ(χ(x, cell_coord)))[local_node_ids[i,j]]    
  end
  0.25*vec(res) 
end

function ιₖ¹(x, Ωc::CoarseTriangulation, global_local_ids)
  cell_coords = Gridap.Geometry.get_cell_coordinates(Ωc.trian); 
  global_elem_ids, local_node_ids =  global_local_ids
  M, N = size(global_elem_ids)
  res = zeros(Float64, M, N)  
  for j=1:N, i=1:M
    cell_coord = cell_coords[global_elem_ids[i,j]]     
    res[i,j] += (ϕᵣ¹(χ(x, cell_coord)))[local_node_ids[i,j]]    
  end
  0.25*vec(res) 
end

"""
Return the cell-wise lazy array of (1-Cˡ)ιₖ
"""
function Cˡιₖ(Ωms::MultiScaleTriangulation, p::Int64, Uh::FESpace, fine_scale_matrices, domain, A)
  # Coarse Scale
  Ωc = Ωms.Ωc
  coarse_trian = Ωc.trian  
  num_coarse_cells = num_cells(coarse_trian)  
  n_monomials = (p+1)^2
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  coarse_dof_vec = lazy_map(Broadcasting(elem_to_dof), get_coarse_scale_patch_coarse_elem_ids(Ωms));
  coarse_dofs_mat = BroadcastVector(combinedims, coarse_dof_vec)
  coarse_dofs = BroadcastVector(vec, coarse_dofs_mat)    
  
  # Fine Triangulation
  Ωf = Ωms.Ωf.trian
  σ_fine = Gridap.Geometry.get_cell_node_ids(Ωf) |> vec
  num_fine_cells = num_cells(Ωf)  
    
  # 1) Get the element-wise map between the coarse and fine-scales
  nsteps = (num_fine_cells/num_coarse_cells) |> sqrt |> log2 |> Int64
  coarse_to_fine_elems = coarsen(num_fine_cells, nsteps) |> vec

  elem_fine = lazy_map(Broadcasting(Reindex(coarse_to_fine_elems)), 1:num_coarse_cells)
  elem_fine_node_ids = lazy_map(Broadcasting(Reindex(σ_fine)), elem_fine)  

  Qₕ = CellQuadrature(Ωf, 4);
  du = get_trial_fe_basis(Uh); dv = get_fe_basis(Uh);
  iwq = ∫( A*∇(du) ⊙ ∇(dv) )Qₕ
  
  # Extract the fine scale matrices
  K, L = fine_scale_matrices
  Ks = lazy_fill(K, num_coarse_cells);
  Ls = lazy_fill(L, num_coarse_cells);  

  patch_interior_fine_scale_dofs = get_coarse_scale_patch_fine_scale_interior_node_indices(Ωms)                                         
    
  P = lazy_fill(patch_interior_fine_scale_dofs, num_coarse_cells)
  C = lazy_fill(coarse_dofs, num_coarse_cells)

  # Extract the iota function only on the coarse    
  nds_fine = vec(Gridap.Geometry.get_node_coordinates(Ωf))


  function I1(nds_fine, Ωc, global_local_ids) 
    res = ιₖ.(nds_fine, Ref(Ωc), Ref(global_local_ids))
    combinedimsview(res,1)  
  end

  function I2(nds_fine, Ωc, global_local_ids) 
    res = ιₖ¹.(nds_fine, Ref(Ωc), Ref(global_local_ids))
    combinedimsview(res,1)  
  end

  global_local_ids = BroadcastVector(find_elements_in_patch, Ref(Ωc), 1:num_coarse_cells, Ref(domain))  

  iota_vals1 = BroadcastVector(I1, Ref(nds_fine), Ref(Ωc), global_local_ids)
  iota_vals2 = BroadcastVector(I2, Ref(nds_fine), Ref(Ωc), global_local_ids)     
                              
  _get_elem_contribs(X, inds) = X[inds]

  elem_wise_contribs = BroadcastVector(_get_elem_contribs, Ref(iwq), elem_fine)
  elem_wise_stima = BroadcastVector(_my_assembler, elem_wise_contribs, elem_fine_node_ids, Ks)                              
  
  Ω_ms = lazy_fill(Ωms, num_coarse_cells)
  domains = lazy_fill(domain, num_coarse_cells)
  BroadcastVector(_compute_stabilization_term, Ks, Ls, P, C, 
                  Ref(elem_wise_stima), Ω_ms, 1:num_coarse_cells, 
                  domains, iota_vals1, iota_vals2)
end

function _my_assembler(cell_vals, conn_matrix, stima)  
  res = zero(stima)
  for i=1:lastindex(cell_vals)
    res[conn_matrix[i], conn_matrix[i]] += cell_vals[i]
  end
  res
end

"""
Compute (1-Cˡ)ιₖ function
"""
function _compute_stabilization_term(stima, lmat, patch_interior_dofs, 
                        coarse_dofs, elem_wise_stima, Ωms, el_ind, domain, iota1, iota2)
  Ωc = Ωms.Ωc
  fine_trian = Ωms.Ωf.trian  
  nds_fine = vec(Gridap.Geometry.get_node_coordinates(fine_trian))
  
  elems_in_patch = find_elements_in_patch(Ωc, el_ind, domain)[1]
  
  elems_in_patch = vec(elems_in_patch)  
  n = size(elems_in_patch,1)
  β = spzeros(length(nds_fine))  
  
  for i=1:n
    el = elems_in_patch[i]     
    
    # Extract the submatrices from the global matrices
    patch_stima = stima[patch_interior_dofs[el], patch_interior_dofs[el]]
    patch_lmat = lmat[patch_interior_dofs[el], coarse_dofs[el]]  
    
    # Compute the right hand side of the corrector problem     
    loadvec = -elem_wise_stima[el]*iota1[:,i] 
    
    # Solve the saddle point system
    Z = spzeros(length(coarse_dofs[el]), length(coarse_dofs[el]))      
    LHS = [patch_stima patch_lmat; patch_lmat' Z]
    RHS = [loadvec[patch_interior_dofs[el]]; zeros(length(coarse_dofs[el]))]      
    SOL = LHS\RHS      
    
    # Compute (-Cˡ)ι = Σₖ (-Cˡₖ)ι
    β[patch_interior_dofs[el]] += SOL[1:length(patch_interior_dofs[el])]      
  end
  # Add the iota function to get (1-Cˡ)ι  
  β += sum(iota2, dims=2)[:,1]
  res = spzeros(Float64, size(stima,1), num_cells(Ωc.trian))
  res[:, el_ind] = β
  res
end

"""
Return the cell-wise lazy array of (1-Cˡ)Pₕνₖ
"""
function Cˡνₖ(β, Ωms::MultiScaleTriangulation, p::Int64)
  # β = get_basis_functions(Vms)
  Ωc = Ωms.Ωc;
  Cmap = x->_c2d(Ωc, x, p);
  lazy_C = lazy_map(Cmap, 1:num_cells(Ωc.trian))  
  num_coarse_cells = num_cells(Ωc.trian) 
  patch_coarse_elems = BroadcastVector(get_patch_coarse_elem, 
                      lazy_fill(Ωc.trian, num_coarse_cells), 
                      lazy_fill(Ωc.tree, num_coarse_cells), 
                      lazy_fill(1, num_coarse_cells), 
                      1:num_coarse_cells);         
  BroadcastVector(_c_times_basis, lazy_C, lazy_fill(β, num_coarse_cells), 
                  patch_coarse_elems, lazy_fill(p,  num_coarse_cells), 
                  1:num_coarse_cells)  
end

"""
Compute (1-Cˡ)Pₕνₖ = ∑ₖ ∑ⱼ (cⱼ,ₖ Λ̃ⱼ,ₖ)
"""
function _c_times_basis(C, β, patch_coarse_elems, p, el)
  βi = β[patch_coarse_elems]
  num_dofs = size(βi[1],2)    
  n_monomials = (p+1)^2
  num_cells = Int64(num_dofs/n_monomials)
  γ = spzeros(Float64, size(βi[1],1), num_cells)  
  elem_to_dof(x) = n_monomials*x-n_monomials+1:n_monomials*x;
  for i=1:lastindex(βi)
    patch_coarse_elem = patch_coarse_elems[i]    
    γ[:, el] += βi[i][:, elem_to_dof(patch_coarse_elem)]*C[:,i]
  end
  γ
end

"""
Coefficients for the νₖ functions
"""
function _c2d(Ωc, el, p; T=Float64)
  num_coarse_cells = num_cells(Ωc.trian) 
  patch_coarse_elems = BroadcastVector(get_patch_coarse_elem, 
                      lazy_fill(Ωc.trian, num_coarse_cells), 
                      lazy_fill(Ωc.tree, num_coarse_cells), 
                      lazy_fill(1, num_coarse_cells), 
                      1:num_coarse_cells);  
  cell_coords = Gridap.Geometry.get_cell_coordinates(Ωc.trian);
  domain = (cell_coords[1][1][1], cell_coords[end][end][1], cell_coords[1][1][2], cell_coords[end][end][2])
  cell_coord = cell_coords[el] 
  Tᵦ⁺ = [-T(1/2) -T(1/2); 
         -T(1/6) T(1/6)] 
  Tᵦ⁻ = [-T(1/2) -T(1/2); 
        -T(1/6) T(1/6)]
  Tᵢ = [-T(1/2) T(-1) -T(1/2); 
        -T(1/6) T(0) T(1/6)]  
  Kᵦ⁰ = [T(2) T(0); 
        T(0) T(0)]
  Kᵦᴺ = [T(0) T(2); 
        T(0) T(0)]
  Kᵢ = [T(0) T(2) T(0); 
        T(0) T(0) T(0)]
  if(length(patch_coarse_elems[el]) == 6)     
    x_coord_1 = cell_coord[1][1]; x_coord_4 = cell_coord[4][1]
    y_coord_2 = cell_coord[2][2]; y_coord_3 = cell_coord[3][2]
    if(x_coord_1 == domain[1]) # LEFT
      C₁¹ = Kᵦ⁰;  C₁² = Tᵦ⁺
      C₂¹ = Kᵢ;  C₂² = Tᵢ
    elseif(x_coord_4 == domain[2]) # RIGHT
      C₁¹ = Kᵦᴺ; C₁² = Tᵦ⁻
      C₂¹ = Kᵢ; C₂² = Tᵢ
    elseif(y_coord_2 == domain[3]) # BOTTOM 
      C₁¹ = Kᵢ; C₁² = Tᵢ
      C₂¹ = Kᵦ⁰; C₂² = Tᵦ⁺
    elseif(y_coord_3 == domain[4]) # TOP
      C₁¹ = Kᵢ; C₁² = Tᵢ
      C₂¹ = Kᵦᴺ; C₂² = Tᵦ⁻
    end
  elseif(length(patch_coarse_elems[el]) == 4)    
    x_coord_1, y_coord_1 = cell_coord[1]; x_coord_4, y_coord_4 = cell_coord[4]
    x_coord_2, y_coord_2 = cell_coord[2]; x_coord_3, y_coord_3 = cell_coord[3]    
    if((x_coord_1,y_coord_1)==(domain[1],domain[3])) # LEFT
      C₁¹ = Kᵦ⁰; C₁² = Tᵦ⁺
      C₂¹ = Kᵦ⁰; C₂² = Tᵦ⁺     
    elseif((x_coord_2,y_coord_2)==(domain[2],domain[3])) # RIGHT
      C₁¹ = Kᵦᴺ; C₁² = Tᵦ⁻
      C₂¹ = Kᵦ⁰; C₂² = Tᵦ⁺
    elseif((x_coord_3,y_coord_3)==(domain[1],domain[4])) # BOTTOM
      C₁¹ = Kᵦ⁰; C₁² = Tᵦ⁺  
      C₂¹ = Kᵦᴺ; C₂² = Tᵦ⁻
    elseif((x_coord_4,y_coord_4)==(domain[2],domain[4])) # TOP      
      C₁¹ = Kᵦᴺ; C₁² = Tᵦ⁻
      C₂¹ = Kᵦᴺ; C₂² = Tᵦ⁻
    end
  else
    C₁¹ = C₂¹ = Kᵢ
    C₁² = C₂² = Tᵢ
  end  
  if(p>=2)
    Z₁¹ = spzeros(T, p-1, size(C₁¹,2))
    Z₂¹ = spzeros(T, p-1, size(C₂¹,2))
    Z₁² = spzeros(T, p-1, size(C₁²,2))
    Z₂² = spzeros(T, p-1, size(C₂²,2))    
    C₁¹ = [C₁¹; Z₁¹]
    C₂¹ = [C₂¹; Z₂¹]
    C₁² = [C₁²; Z₁²]
    C₂² = [C₂²; Z₂²]
  end         
  res = kron(C₂¹, C₁¹) - kron(C₂², C₁²)  
  res[1:(p+1)*(p+1),:]/length(patch_coarse_elems[el])
end

"""
The stabilized multiscale finite element space in 2d.

StabilizedMultiScaleFESpace(Ωms::MultiScaleTriangulation, 
                            q::Int64, p::Int64, A::CellField, 
                            qorder::Int64)
"""
struct StabilizedMultiScaleFESpace <: FESpace
  Ω::MultiScaleTriangulation
  order::Int64
  Uh::FESpace
  basis_vec_ms
  fine_scale_system
end

function StabilizedMultiScaleFESpace(Vms::T, p::Int64, Uh::FESpace, fine_scale_matrices, domain::NTuple{4,Float64}, A) where T<:FESpace
  Ωms = Vms.Ω
  α = get_basis_functions(Vms)
  K, L = fine_scale_matrices  
  β = Cˡιₖ(Ωms, p, Uh, (K, L), domain, A);
  γ = Cˡνₖ(α, Ωms, p);    
  δ = copy(α)
  num_coarse_cells = num_cells(Ωms.Ωc.trian)
  basis_vec_ms = BroadcastVector(_replace_new_basis!, δ, β + γ, p, 1:num_coarse_cells)
  StabilizedMultiScaleFESpace(Ωms, p, Uh, basis_vec_ms, fine_scale_matrices)
end

function _replace_new_basis!(α, γ, p, i)
  elem_to_dof(x) = (p+1)^2*x-(p+1)^2+1
  α[:, elem_to_dof(i)] = γ[:, i]  
  α
end

import MultiscaleFEM.MultiscaleBases: get_basis_functions
get_basis_functions(V::StabilizedMultiScaleFESpace) = V.basis_vec_ms

end