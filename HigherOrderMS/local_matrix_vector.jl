#########################################################
# Functions to generate the local matrix-vector systems
# Does not depend on the order of polynomial
#########################################################

"""
Function to get the local matrix-vector system corresponding to the inner products:
    (u,v) = ∫ₖ u*v dx
    (f,v)  = ∫ₖ f*v dx
Here u,v ∈ H¹₀(K) and f is a known function.
"""
function _local_matrix!(Me::Array{Float64}, basis::Tuple{Vector{Float64},Vector{Float64}}, 
  A::Float64, w::Float64, h::Float64, fespace::Tuple{Int64,Int64})
  q,p = fespace
  basis_1, basis_2 = basis
  # Loop over the local matrices
  for j=1:p+1, i=1:q+1
    Me[i,j] += A*w*basis_1[i]*basis_2[j]*(0.5*h)
  end
end
function _local_vector!(Fe::Array{Float64}, basis::Vector{Float64}, A::Float64, w::Float64, h::Float64, fespace::Int64)
  p = fespace
  for i=1:p+1
    Fe[i] += A*w*basis[i]*(0.5*h)
  end
end
