#### ##### ##### ##### ##### ##### ##### ##### ##### ###
# Module containing the MultiScaleBases implementation #
#### ##### ##### ##### ##### ##### ##### ##### ##### ###

module MultiScaleBases
  include("basis-functions.jl")
  include("assemble_matrices.jl")
  include("solve.jl")

  using SparseArrays
  using LoopVectorization

  function compute_ms_bases!(cache, p::Int64, l::Int64)
    mats, fvecs, bases_data = cache
    f1, f2 = fvecs
    stima, lmat = mats
    basis_vec_ms, patch_to_fine_scale = bases_data
    nc = size(basis_vec_ms,1)
    index = 1
    for t=1:nc
      start = max(1,t-l)
      last = min(nc,t+l)
      gn = patch_to_fine_scale[t]
      fn = 2:length(gn)-1
      stima_el = stima[gn[fn], gn[fn]]
      lmat_el = lmat[gn[fn], start*(p+1)-p:last*(p+1)]
      for tt=1:p+1
        fvecs_el = vcat(f1[gn[fn],index], f2[start*(p+1)-p:last*(p+1), index])
        lhs = [stima_el lmat_el; (lmat_el)' spzeros(Float64,size(lmat_el,2), size(lmat_el,2))]
        rhs = fvecs_el                      
        basis_vec_ms[t][gn[fn],tt] = (lhs\rhs)[1:size(stima_el,1)]
        index += 1   
      end
    end
    basis_vec_ms
  end

  function ms_basis_cache!(matcache, nds::Tuple{AbstractVector{Float64}, AbstractVector{Float64}}, 
    elem::Tuple{Matrix{Int64}, Matrix{Int64}},
    quad::Tuple{Vector{Float64}, Vector{Float64}}, fespaces::Tuple{Int64,Int64}, 
    D::Function, basis_vec_ms::Vector{Matrix{Float64}}, patch_to_fine_scale::Vector{AbstractVector{Int64}})
    
    elem_coarse, elem_fine = elem
    nds_coarse, nds_fine = nds
    nf = size(elem_fine,1)
    nc = size(elem_coarse,1)
    q,p = fespaces
    stima_cache, l_mat_cache, l2_mat_cache = matcache
    
    stima = AssembleMatrices.assemble_matrix!(stima_cache, D, StandardBases.∇φᵢ!, StandardBases.∇φᵢ!, -1)
    lmat = AssembleMatrices.assemble_lm_matrix!(l_mat_cache, StandardBases.Λₖ!, StandardBases.φᵢ!, 1)
    fvecs = AssembleMatrices.assemble_lm_l2_matrix!(l2_mat_cache, StandardBases.Λₖ!, StandardBases.Λₖ!, 1) 
    
    (stima, lmat), (zeros(Float64, q*nf+1, nc*(p+1)), fvecs), (basis_vec_ms, patch_to_fine_scale)
  end
end