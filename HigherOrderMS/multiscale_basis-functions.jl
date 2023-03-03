#### ##### ##### ##### ##### ##### ##### ##### ##### ###
# Module containing the MultiScaleBases implementation #
#### ##### ##### ##### ##### ##### ##### ##### ##### ###

module MultiScaleBases
  include("basis-functions.jl")
  include("assemble_matrices.jl")
  include("solve.jl")

  using SparseArrays
  using LoopVectorization

  function compute_ms_bases!(cache, Λ!::Function, u!::Function, J_exp::Int64)
    mats, fvec = bases_data
    stima, lmat = mats
    basis_vec_ms, patch_to_fine_scale = bases_data
    nc = size(basis_vec_ms,1)
    for t=1:nc
      nds = (nds_elem[t,1], nds_elem[t,2])
      stima_el = stima[patch_to_fine_scale[t], patch_to_fine_scale[t]]
      lmat_el = lmat[patch_to_fine_scale[t], :]
    end
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
    fn = 2:q*nf
    qs, ws = quad
    stima_cache, l_mat_cache, l2_mat_cache = matcache
    
    stima = AssembleMatrices.assemble_matrix!(stima_cache, D, StandardBases.∇φᵢ!, StandardBases.∇φᵢ!, -1)
    lmat = AssembleMatrices.assemble_lm_matrix!(l_mat_cache, StandardBases.Λₖ!, StandardBases.φᵢ!, 1)
    fvecs = AssembleMatrices.assemble_lm_l2_matrix!(l2_mat_cache, StandardBases.Λₖ!, StandardBases.Λₖ!, 1)  
    
    (stima, lmat), fvecs, (nds_coarse[elem_coarse], basis_vec_ms, patch_to_fine_scale)
  end
end