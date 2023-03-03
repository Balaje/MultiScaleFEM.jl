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
    basis_vec_ms, fns = bases_data
    nc = size(basis_vec_ms,1)
    index = 1
    for t=1:nc
      start = max(1,t-l)
      last = min(nc,t+l)
      fn = fns[t]
      stima_el = stima[fn, fn]
      lmat_el = lmat[fn, start*(p+1)-p:last*(p+1)]
      for tt=1:p+1
        fvecs_el = vcat(f1[fn,index], f2[start*(p+1)-p:last*(p+1), index])
        lhs = [stima_el lmat_el; (lmat_el)' spzeros(Float64,size(lmat_el,2), size(lmat_el,2))]
        rhs = fvecs_el                      
        basis_vec_ms[t][fn,tt] = (lhs\rhs)[1:size(stima_el,1)]
        index += 1   
      end
    end
    basis_vec_ms
  end

  function ms_basis_cache!(matcache, nf::Int64, nc::Int64, fespaces::Tuple{Int64,Int64}, 
    D::Function, basis_vec_ms::Vector{Matrix{Float64}}, patch_to_fine_scale::Vector{AbstractVector{Int64}})
    q,p = fespaces
    stima_cache, l_mat_cache, l2_mat_cache = matcache
    stima = AssembleMatrices.assemble_matrix!(stima_cache, D, StandardBases.∇φᵢ!, StandardBases.∇φᵢ!, -1)
    lmat = AssembleMatrices.assemble_lm_matrix!(l_mat_cache, StandardBases.Λₖ!, StandardBases.φᵢ!, 1)
    fvecs = AssembleMatrices.assemble_lm_l2_matrix!(l2_mat_cache, StandardBases.Λₖ!, StandardBases.Λₖ!, 1) 
    fns = [gn[2:length(gn)-1] for gn in patch_to_fine_scale]
    (stima, lmat), (zeros(Float64, q*nf+1, nc*(p+1)), fvecs), (basis_vec_ms, fns)
  end
end