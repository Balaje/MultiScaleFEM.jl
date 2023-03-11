##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #
#  Functions to assemble the stiffness, mass and multiscale matrix vector system   #
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #

function get_saddle_point_problem(domain::Tuple{Float64,Float64}, D::Function, f::Function, 
  fespaces::Tuple{Int64,Int64}, nels::Tuple{Int64,Int64}, qorder::Int64)
  nf, nc = nels
  q, p = fespaces
  model = CartesianDiscreteModel(domain, (nf,))
  reffe = ReferenceFE(lagrangian,Float64,q)
  U = TestFESpace(model,reffe,conformity=:H1)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qorder)
  # The main system.
  a(u,v) = ∫(D*(∇(v))⊙(∇(u)))dΩ
  c(u,v) = ∫((v)⋅(u))dΩ
  l(v) = ∫(f*v)dΩ
  op = AffineFEOperator(a,l,U,U)   
  K = op.op.matrix
  F = op.op.vector
  op = AffineFEOperator(c,l,U,U)
  M = op.op.matrix
  # The rectangular matrix
  elem_coarse = [i+j for i=1:nc, j=0:p]
  nds_coarse = LinRange(domain[1], domain[2], nc+1)
  L = spzeros(Float64, size(K,1), nc*(p+1))
  index = 1
  for t=1:nc
    nds = (nds_coarse[elem_coarse[t,1]], nds_coarse[elem_coarse[t,2]])
    for j=1:p+1    
      LP(y) = Λₖ!(y[1], nds, p, j)
      m(v) = ∫(LP*v)dΩ
      op = AffineFEOperator(a,m,U,U)
      L[:,index] = op.op.vector
      index += 1
    end
  end
  # The L² Projection of the Legendre basis
  Λ = assemble_lm_l2_matrix(nds_coarse, elem_coarse, p)
  (K,M,F), L, Λ, U
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
    @simd for i=1:p+1
      @inbounds l2mat[(index-1)+i, (index-1)+i] = (h/(2*(i-1)+1))
    end
    index = index + (p+1)
  end
  l2mat
end

function assemble_ms_matrix(ms_elem_mats, ms_elem::Vector{Vector{Int64}}, nc::Int64, p::Int64)
  ijv = BroadcastVector(findnz, ms_elem_mats);
  i = BroadcastVector(repeat, ms_elem, BroadcastVector(length, ms_elem))
  j = BroadcastVector(vec, BroadcastVector(repeat, BroadcastVector(transpose, ms_elem), BroadcastVector(length, ms_elem)));
  v = BroadcastVector(getindex, ijv, 3);
  M = BroadcastVector(sparse, i, j, v, nc*(p+1), nc*(p+1));
  applied(sum, applied(Tuple, M))
end
function assemble_ms_vector(ms_elem_vecs, ms_elem::Vector{Vector{Int64}}, nc::Int64, p::Int64)
  v = BroadcastVector(sparsevec, ms_elem, ms_elem_vecs, nc*(p+1))
  applied(sum, applied(Tuple, v))
end

function get_solution(sol, basis_vecs::SparseMatrixCSC{Float64,Int64})  
  nc = size(sol, 1)
  p = Int(size(basis_vecs, 2)/nc)-1
  ndof = size(basis_vecs,1)
  res = Vector{Float64}(undef, ndof)
  fill!(res, 0.0)
  index = 1
  for j=1:nc, i=0:p
    @simd for tt=1:ndof
      @inbounds res[tt] += sol[(p+1)*j+i-p]*basis_vecs[tt,index]      
    end
    index+=1
  end
  res
end