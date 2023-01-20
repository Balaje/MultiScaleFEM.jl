###########################################
# Program to solve the Multiscale problem.
###########################################

include("eg1.jl"); # Contains the basis functions

# Define the assembler for the MS-space
struct MultiScaleSpace <: Strategy end
function MatrixAssembler(x::MultiScaleSpace, fespace::Int64, elem::Matrix{Int64}, l::Int64)
  new_elem = _new_elem_matrices(elem, fespace, x, l)
end 
function _new_elem_matrices(elem, fespace, l, ::MultiScaleSpace)
  N = size(elem,1)
  p = fespace
  l2elems = _new_elem_matrices(elem, p, L²ConformingSpace())
  elems = Matrix{Int64}(undef, N, (p+1)*(l+2))
  fill!(elems,0)
  for el=1:N
    start = (el-l)<1 ? 1 : el-l; last = start+2l
    last = (last>n) ? n : last; start = last-2l
    #elems[i,:] = start:last
    @show start:last, l2elems[start,1]:l2elems[last,2]
  end
  elems
end 

#new_elem = _new_elem_matrices(Ω.elems, p, l, MultiScaleSpace())