module Assemblers

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData

using SparseArrays
using SplitApplyCombine
using ProgressMeter

using MultiscaleFEM.CoarseToFine: get_fine_nodes_in_coarse_elems

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
function assemble_rect_matrix(coarse_trian::Triangulation, fine_scale_space::FESpace, local_to_global_map, p::Int64)
  fine_trian = get_triangulation(fine_scale_space)
  M = assemble_massma(fine_scale_space, x->1.0, p)  
  L = legendre_basis_on_fine_scale(coarse_trian, fine_trian, local_to_global_map, p)
  M*L
end

function legendre_basis_on_fine_scale(coarse_trian::Triangulation, fine_trian::Triangulation, local_to_global_map, p::Int64)  
  # Coarse scale
  num_coarse_cells = num_cells(coarse_trian)  
  coarse_cell_coords = get_cell_coordinates(coarse_trian)  
  n_monomials = (p+1)^2
  # Fine Scale  
  fine_node_coords = vec(get_node_coordinates(fine_trian))  
  # Map between coarse and fine scale
  fine_node_indices_in_coarse_elem, fine_node_coords_in_coarse_elems = get_fine_nodes_in_coarse_elems(local_to_global_map, fine_node_coords)  
  # Compute the matrix
  L = spzeros(Float64, num_nodes(fine_trian), num_coarse_cells*n_monomials)        
  for i=1:num_coarse_cells
    c = coarse_cell_coords[i]
    L[fine_node_indices_in_coarse_elem[i], (i-1)*(p+1)^2+1:i*(p+1)^2] = combinedims(_2d_legendre_polynomial_on_coarse_cell.(fine_node_coords_in_coarse_elems[i], Ref(c), Ref(p)))'
  end
  L
end

function _2d_legendre_polynomial_on_coarse_cell(x, coarse_cell_coord, p)
  nds_x = (coarse_cell_coord[1][1], coarse_cell_coord[2][1])
  nds_y = (coarse_cell_coord[1][2], coarse_cell_coord[3][2])  
  αβ = Assemblers.poly_exps(p)
  n_monomials = (p+1)^2
  res = zeros(Float64, n_monomials)
  for i=1:n_monomials
    res[i] = Λₖ(x, nds_x, nds_y, p, αβ[i])
  end
  res
end

"""
Function to assemble the inner-product of the L² functions.
"""
function assemble_rhs_matrix(coarse_trian::Triangulation, p::Int64)
  # Coarse cells
  num_patch_coarse_cells = num_cells(coarse_trian)
  n_monomials = (p+1)^2  
  coarse_cell_coords = get_cell_coordinates(coarse_trian)   
  # Compute the matrix
  L = spzeros(Float64, num_patch_coarse_cells*n_monomials, num_patch_coarse_cells*n_monomials)  
  index = 1
  for i=1:num_patch_coarse_cells
    coarse_cell_coord = coarse_cell_coords[i]
    nds_y = (coarse_cell_coord[1][2], coarse_cell_coord[3][2])
    h = (nds_y[2]-nds_y[1])
    for j=1:n_monomials
      L[index+j-1, index+j-1] = (h/(2*(j-1)+1))*(h/(2*(j-1)+1))
    end
    index += n_monomials
  end
  L
end

"""
Scaled monomial bases at each coarse rectanles. 
The L² functions in the higher order MS Method is the tensor product of the Legendre polynomials in the cell
"""
function Λₖ(x::Point, nds_x::NTuple{2,Float64}, nds_y::NTuple{2,Float64}, p::Int64, αβ::NTuple{2,Int64})  
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
assemble_ms_matrix(B::Vector{SparseMatrix}, K::SparseMatrix, p::Int64):
1) B - Multiscale bases on the coarse scale.
2) K - Fine-scale matrix (eg. stiffness/mass)
3) p - Polynomial approximation order

Function to assemble the multiscale matrix corresponding to the fine-scale matrix K
"""
function assemble_ms_matrix(B, K)  
  B'*K*B
end

"""
assemble_ms_loadvec(B::Vector{SparseMatrix}, F::Vector, p::Int64):
1) B - Multiscale bases on the coarse scale.
2) F - Fine-scale vector (eg. load vector)
3) p - Polynomial approximation order

Function to assemble the multiscale matrix corresponding to the fine-scale matrix K
"""
function assemble_ms_loadvec(B, F)
  B'*F
end

"""
Function to convert the multiscale matrix to the fine scale.
"""
function assemble_fine_scale_from_ms(V)
  n_coarse_scale = length(V)
  res = zero(V[1])
  for i=1:n_coarse_scale
    res += V[i]
  end
  res
end

"""
Function to solve the multiscale problem and split the dimensions to convert to fine scale.
"""
function solve_ms_problem(K, F, dims)
  sol = K\F  
  splitdimsview(reshape(sol, dims))
end

end