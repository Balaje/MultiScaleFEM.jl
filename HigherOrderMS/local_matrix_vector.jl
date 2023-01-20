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
function _local_matrix!(Me::Array{Float64}, xn::VecOrMat{Float64}, basis::Tuple{Function,Function}, A::Function, 
  quad::Tuple{Vector{Float64},Vector{Float64}}, h::Float64, fespace::Tuple{Int64,Int64})
  fill!(Me, 0.)
  qs,ws = quad
  q,p = fespace
  basis_1, basis_2 = basis
  for qk=1:lastindex(qs)
    x̂ = qs[qk]
    x = (xn[2]+xn[1])*0.5 .+ 0.5*h*x̂
    # Loop over the local matrices
    for i=1:q+1, j=1:p+1
      Me[i,j] += ws[qk]*( A(x) * basis_1(x)[i] * basis_2(x)[j] )*(0.5*h)
    end
  end
end
function _local_vector!(Fe::Array{Float64}, xn::VecOrMat{Float64}, basis::Function, f::Function, 
  quad::Tuple{Vector{Float64}, Vector{Float64}}, h::Float64, fespace::Int64)
  fill!(Fe, 0.)
  qs,ws = quad
  p = fespace
  for q=1:lastindex(qs)
    x̂ = qs[q]
    x = (xn[2]+xn[1])*0.5 .+ 0.5*h*x̂
    for i=1:p+1
      Fe[i] += ws[q]*( f(x) * basis(x)[i] )*(0.5*h)
    end
  end
end
