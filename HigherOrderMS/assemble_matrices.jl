##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #
#  Functions to assemble the stiffness, mass and multiscale matrix vector system   #
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #

struct FineScaleSpace
  domain::Tuple
  nf::Int64
  q::Int64
  U::FESpace
  dΩ::Measure
  assem::SparseMatrixAssembler
end
function FineScaleSpace(domain::Tuple{Float64,Float64}, q::Int64, qorder::Int64, nf::Int64)
  model = CartesianDiscreteModel(domain, (nf,))
  reffe = ReferenceFE(lagrangian, Float64, q)
  U = TestFESpace(model, reffe, conformity=:H1)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qorder)
  assem = SparseMatrixAssembler(U,U)
  FineScaleSpace(domain, nf, q, U, dΩ, assem)
end 

function get_saddle_point_problem(fspace::FineScaleSpace, D::Function, p::Int64, nc::Int64)
  # The stiffness matrix
  domain = fspace.domain
  K = assemble_stiffness_matrix(fspace, D)
  # The rectangular matrix
  elem_coarse = [i+j for i=1:nc, j=0:p]
  nds_coarse = LinRange(domain[1], domain[2], nc+1)
  L = spzeros(Float64, size(K,1), nc*(p+1))
  index = 1
  for t=1:nc
    nds = (nds_coarse[elem_coarse[t,1]], nds_coarse[elem_coarse[t,2]])
    for j=1:p+1    
      LP(y) = Λₖ!(y[1], nds, p, j)
      L[:,index] = assemble_load_vector(fspace, LP)
      index += 1
    end
  end
  # The L² Projection of the Legendre basis
  Λ = assemble_lm_l2_matrix(nds_coarse, elem_coarse, p)
  K, L, Λ
end

function assemble_stiffness_matrix(fspace::FineScaleSpace, D::Function)
  U = fspace.U
  assem = fspace.assem
  dΩ = fspace.dΩ
  # The main system.
  a(u,v) = ∫(D*(∇(v))⊙(∇(u)))dΩ
  assemble_matrix(a, assem, U, U)
end
function assemble_load_vector(fspace::FineScaleSpace, f::Function)
  U = fspace.U
  assem = fspace.assem
  dΩ = fspace.dΩ
  # The main system.
  l(v) = ∫(f*v)dΩ
  assemble_vector(l, assem, U)
end
function assemble_mass_matrix(fspace::FineScaleSpace, c::Function)
  U = fspace.U
  assem = fspace.assem
  dΩ = fspace.dΩ
  # The main system.
  m(u,v) = ∫(c*(v)⋅(u))dΩ 
  assemble_matrix(m, assem, U, U)
end

"""
Function to assemble the matrix associated with the RHS of the saddle point problem
"""
function assemble_lm_l2_matrix(nds::AbstractVector{Float64}, elem::Matrix{Int64}, p::Int64)   
  nc = size(elem,1)  
  l2mat = Diagonal(ones(Float64,nc*(p+1)))
  index = 1
  for t=1:nc
    h = nds[elem[t,2]] - nds[elem[t,1]]
    for i=1:p+1
      l2mat[(index-1)+i, (index-1)+i] = (h/(2*(i-1)+1))
    end
    index = index + (p+1)
  end
  l2mat
end