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
function assemble_rect_matrix(coarse_trian::Triangulation, fine_space::FESpace, p::Int64)  
  # Coarse Scale
  num_patch_coarse_cells = num_cells(coarse_trian)
  coarse_cell_coords = get_cell_coordinates(coarse_trian)
  # Fine Scale
  Ω_fine = get_triangulation(fine_space)    
  fine_node_coordinates = vec(get_node_coordinates(Ω_fine))
  n_monomials = (p+1)^2
  L = spzeros(Float64, num_nodes(Ω_fine), num_patch_coarse_cells*n_monomials)      
  M = assemble_massma(fine_space, x->1.0, 0)
  # Begin computing the DOF matrices
  index = 1
  αβ = poly_exps(p)
  for i=1:num_patch_coarse_cells
    coarse_cell_coord = coarse_cell_coords[i]
    nds_x = (coarse_cell_coord[1][1], coarse_cell_coord[2][1])
    nds_y = (coarse_cell_coord[1][2], coarse_cell_coord[3][2])    
    for j=1:n_monomials
      b = x->ℳ(x, nds_x, nds_y, p, αβ[j])          
      L[:,index] = b.(fine_node_coordinates)      
      index = index+1
    end
  end
  M*L
end

"""
Function to assemble the inner-product of the L² functions.
"""
function assemble_rhs_matrix(coarse_trian::Triangulation, p::Int64)
  node_coords = get_node_coordinates(coarse_trian)
  cell_ids = get_cell_node_ids(coarse_trian)
  coarse_cell_coords = lazy_map(Broadcasting(Reindex(node_coords)), cell_ids)
  num_patch_coarse_cells = num_cells(coarse_trian)
  n_monomials = (p+1)^2
  L = spzeros(Float64, num_patch_coarse_cells*n_monomials, num_patch_coarse_cells*n_monomials)  
  index = 1
  coarse_cell_coords = get_cell_coordinates(coarse_trian)  
  coarse_cell_coords = get_cell_coordinates(coarse_trian)  
  αβ = poly_exps(p)
  coarse_cell_coords = get_cell_coordinates(coarse_trian)    
  αβ = poly_exps(p)
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