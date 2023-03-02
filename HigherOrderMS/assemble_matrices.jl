##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #
#  Functions to assemble the stiffness, mass and multiscale matrix vector system   #
##### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### #
module AssembleMatrices
  include("basis-functions.jl")

  using SparseArrays   
  using LoopVectorization

  function assembler_cache(nds::AbstractVector{Float64}, elem::Matrix{Int64}, 
    quad::Tuple{Vector{Float64}, Vector{Float64}}, q::Int64)
    nds_elem = nds[elem]
    nel = size(elem,1)
    iiM = Vector{Int64}(undef, (q+1)^2*nel)
    jjM = Vector{Int64}(undef, (q+1)^2*nel)
    vvM = Vector{Float64}(undef, (q+1)^2*nel)
    fill!(iiM,0)
    fill!(jjM,0)
    fill!(vvM,0.0)
    iiV = Vector{Int64}(undef,(q+1)*nel)
    vvV = Vector{Float64}(undef,(q+1)*nel)
    fill!(iiV,0)
    fill!(vvV,0.0)
    qs = quad[1]
    xqs = Matrix{Float64}(undef, nel, length(qs))
    J = (nds_elem[:,2]-nds_elem[:,1])*0.5
    for i=1:lastindex(qs)
      xqs[:,i] = (nds_elem[:,2]+nds_elem[:,1])*0.5 + (nds_elem[:,2]-nds_elem[:,1])*0.5*qs[i]
    end
    Dxqs = similar(xqs)
    fill!(Dxqs,0.0)
    bc = MultiScaleBases.lagrange_basis_cache(q)
    h1elem =  [q*i+j-(q-1) for i = 1:nel, j=0:q]
    index = 0
    assem_mat = spzeros(Float64, q*nel+1, q*nel+1)
    assem_vec = zeros(Float64, q*nel+1)
    (xqs, Dxqs, J), (iiM, jjM, vvM), (iiV, vvV), (h1elem, h1elem), (bc,bc), quad, assem_mat, assem_vec
  end

  function assemble_matrix!(cache, D::Function, u!::Function, v!::Function, J_exp::Int64)
    quadData = cache[1]
    matData = cache[2]
    elems = cache[4]
    bu, bv = cache[5] 
    quad = cache[6]
    M = cache[7]
    # Unpack all the data
    qs, ws = quad
    xqs, Dxqs, J = quadData
    iiM, jjM, vvM = matData    
    elem_i, elem_j = elems
    nel = size(elem_i,1)
    p = size(elem_i,2) - 1
    q = size(elem_j,2) - 1
    # Fill up the matrices
    qrule = size(xqs, 2)
    map!(D, Dxqs, xqs)
    fill!(M,0.0)
    fill!(vvM,0.0)
    for k=1:qrule
      xq = qs[k]
      wq = ws[k]
      u!(bu, xq)
      v!(bv, xq)      
      index = 0
      for i=1:p+1, j=1:q+1        
        @turbo for l=1:nel
          iiM[index+l] = elem_i[l, i]
          jjM[index+l] = elem_j[l, j]
          vvM[index+l] += Dxqs[l,k]*(J[l])^J_exp*bu[3][i]*bv[3][j]*wq                              
        end
        index+=nel
      end
    end    
    @inbounds for k=1:lastindex(iiM)
      M[iiM[k], jjM[k]] += vvM[k]
    end
    M
  end
  
  function assemble_vector!(cache, f::Function, u!::Function, J_exp::Int64)
    quadData = cache[1]
    vecData = cache[3]
    elems = cache[4]
    bu = cache[5][1]
    quad = cache[6]
    F = cache[8]
    # Unpack all the data
    qs, ws = quad
    xqs, fxqs, J = quadData
    iiV, vvV = vecData
    elem_i = elems[1]
    nel = size(elem_i,1)
    q = size(elem_i,2) - 1
    # Fill up the matrices
    qrule = size(xqs,2)
    map!(f, fxqs, xqs)
    fill!(F,0.0)
    fill!(vvV,0.0)
    for k=1:qrule
      xq = qs[k]
      wq = ws[k]
      u!(bu, xq)
      index = 0
      for i=1:q+1
        @turbo for l=1:nel
          iiV[index+l] = elem_i[l,i]
          vvV[index+l] += fxqs[l,k]*(J[l])^J_exp*bu[3][i]*wq
        end
        index+=nel
      end
    end
    @inbounds for k=1:lastindex(iiV)
      F[iiV[k]] += vvV[k]
    end
    F
  end
end