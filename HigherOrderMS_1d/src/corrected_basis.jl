module AdditionalCorrections

using ProgressMeter
using SparseArrays

using HigherOrderMS_1d.Assemblers: FineScaleSpace, get_saddle_point_problem, assemble_stiffness_matrix, assemble_mass_matrix

##### ###### ###### ###### ###### ###### ###### ###### #
# Program to implement the corrected basis function
##### ###### ###### ###### ###### ###### ###### ###### #
function compute_additional_correction_basis(fine_scale_space::FineScaleSpace, D::Function, p::Int64, nc::Int64, l::Int64, 
                                  patch_indices_to_global_indices::Vector{AbstractVector{Int64}}, p′::Int64,
                                  basis_vec_ms₁::AbstractMatrix; T=Float64, ntimes=1)
  (ntimes==0) && return T[]
  nf = fine_scale_space.nf
  q = fine_scale_space.q
  basis_vec_ms = spzeros(T, q*nf+1, ntimes*(p′+1)*nc) # To store the multiscale basis functions
  _, L, Λ  = get_saddle_point_problem(fine_scale_space, D, p, nc)
  K = assemble_stiffness_matrix(fine_scale_space, D)
  M = assemble_mass_matrix(fine_scale_space, x->1.0)
  β = basis_vec_ms₁ 
  index = 1
  for corr = 1:ntimes
    index_1 = 1
    @showprogress desc="Computing the additional correction level $corr" for t=1:nc
      fullnodes₁ = patch_indices_to_global_indices[t]    
      bnodes₁ = [fullnodes₁[1], fullnodes₁[end]]        
      freenodes₁ = setdiff(fullnodes₁, bnodes₁)    
      start₁ = max(1,t-l); last₁ = min(nc,t+l)    
      gn₁ = start₁*(p+1)-p:last₁*(p+1)    
      stima_el = K[freenodes₁,freenodes₁]
      lmat_el = L[freenodes₁,gn₁]
      for _=1:p′+1
        fvecs_el = [M[freenodes₁,freenodes₁]*β[freenodes₁,index_1]; zeros(T,length(gn₁))]        
        lhs = [stima_el lmat_el; (lmat_el)'  spzeros(T, length(gn₁), length(gn₁))]
        rhs = fvecs_el           
        sol = lhs\collect(rhs)
        basis_vec_ms[freenodes₁, index] = sol[1:length(freenodes₁)]
        index += 1   
        index_1 += 1
      end
    end
    β = basis_vec_ms[:,1+(corr-1)*((p′+1)*nc):(corr)*(nc*(p′+1))]
  end
  basis_vec_ms
end

end
