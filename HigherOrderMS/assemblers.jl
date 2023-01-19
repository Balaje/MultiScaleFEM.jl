############################################################################
# Contains data structures to generate the assemblers of the inner-products
############################################################################

abstract type Assembler <: Any end
abstract type Strategy <: Any end

struct H¹ConformingSpace <: Strategy end
struct L²ConformingSpace <: Strategy end
"""
mutable struct MatrixAssembler <: Assembler
  iM::Array{Int64}
  jM::Array{Int64}
end

Structure containing the local--global map for assembling the matrices
"""
mutable struct MatrixAssembler <: Assembler
  iM::Array{Int64}
  jM::Array{Int64}
end
function MatrixAssembler(x::A, y::B,
                         fespaces::Tuple{Int64,Int64},
                         elem::Tuple{Matrix{Int64}, Matrix{Int64}}) where {A,B<:Union{H¹ConformingSpace, L²ConformingSpace}}
  new_elem = _new_elem_matrices(elem, fespaces, x, y)
  iM, jM = _get_assembler_matrix(new_elem, fespaces, x, y)
  MatrixAssembler(iM, jM)
end
function MatrixAssembler(x::A, fespace::Int64, elem::Matrix{Int64}) where A<:Union{H¹ConformingSpace, L²ConformingSpace}
  new_elem = _new_elem_matrices(elem, fespace, x)
  iM, jM = _get_assembler_matrix(new_elem, fespace)
  MatrixAssembler(iM, jM)
end
Base.transpose(m::MatrixAssembler) = MatrixAssembler(m.jM, m.iM)
Base.adjoint(m::MatrixAssembler) = MatrixAssembler(m.jM, m.iM)

"""
mutable struct VectorAssembler <: Assembler
  iV::Array{Int64}
end

Structure containing the local--global map for assembling the vectors
"""
mutable struct VectorAssembler <: Assembler
  iV::Array{Int64}
end
function VectorAssembler(x::A, p::Int64, elem::Matrix{Int64}) where {A<:Union{H¹ConformingSpace, L²ConformingSpace}}
  new_elems = _new_elem_matrices(elem, p, x)
  iV = _get_assembler_vector(new_elems, p)
  VectorAssembler(iV)
end

"""
mutable struct MatrixVectorAssembler <: Assembler
  mAssem::MatrixAssembler
  vAssem::VectorAssembler
end
"""
mutable struct MatrixVectorAssembler <: Assembler
  mAssem::MatrixAssembler
  vAssem::VectorAssembler
end
#################################################################
# My assembly strategies: For H¹Conforming and L²Conforming
# - First generate the element matrices for a polynomial of order q
# - Next generate the connectivity matrices for the method
#################################################################

# 1) New element matrices
function _new_elem_matrices(elem::Matrix{Int64}, p::Int64, ::H¹ConformingSpace)
  N = size(elem,1)
  elems = Matrix{Int64}(undef,N,p+1)
  fill!(elems,0)
  for i=1:N
    elems[i,:] = elem[i,1]+(i-1)*(p-1): elem[i,2]+i*(p-1)
  end
  elems
end
function _new_elem_matrices(elem::Matrix{Int64}, p::Int64, ::L²ConformingSpace)
  N = size(elem,1)
  elems = Matrix{Int64}(undef,N,p+1)
  fill!(elems,0)
  for i=1:N
    elems[i,:] = elem[i,1]+(i-1)*(p): elem[i,2]+i*(p)-1
  end
  elems
end
function _new_elem_matrices(elem::Tuple{Matrix{Int64}, Matrix{Int64}},
                            fespace::Tuple{Int64, Int64}, ::H¹ConformingSpace, ::L²ConformingSpace)
  elem_1, elem_2 = elem
  p,q = fespace
  new_elem_1 = _new_elem_matrices(elem_1, p, H¹ConformingSpace())
  new_elem_2 = _new_elem_matrices(elem_2, q, L²ConformingSpace())
  new_elem_1, new_elem_2
end


# Assembler trio for the (H¹Conforming,H¹Conforming)/(L²Conforming, L²Conforming) elements
function get_assembler(elem::Matrix{Int64}, p::Int64)
  iM, jM = _get_assembler_matrix(elem, p)
  iV = _get_assembler_vector(elem, p)
  iM, jM, iV
end
function _get_assembler_matrix(elem::Matrix{Int64}, p::Int64)
  nel =  size(elem,1)
  iM = Array{Int64}(undef, nel, p+1, p+1)
  jM = Array{Int64}(undef, nel, p+1, p+1)
  fill!(iM,0); fill!(jM, 0);
  for t=1:nel, ti=1:p+1, tj=1:p+1
    iM[t,ti,tj] = elem[t,ti]
    jM[t,ti,tj] = elem[t,tj]
  end
  iM, jM
end

# Assembler trio for the (H¹Conforming, L²Conforming) elements
function get_assembler(elem::Tuple{Matrix{Int64}, Matrix{Int64}},
                       fespace::Tuple{Int64, Int64},
                       ::H¹ConformingSpace, ::L²ConformingSpace)
  iM, jM = _get_assembler_matrix(elem, fespace, H¹ConformingSpace(), L²ConformingSpace())
  iV = _get_assembler_vector(elem[2], fespace[2])
  iM, jM, iV
end
function _get_assembler_matrix(elem::Tuple{Matrix{Int64}, Matrix{Int64}},
                               fespace::Tuple{Int64, Int64},
                               ::H¹ConformingSpace, ::L²ConformingSpace)
  q,p = fespace
  elem_1, elem_2 = elem
  nel_1 = size(elem_1,1)
  nel_2 = size(elem_2,1)
  iM = Array{Int64}(undef, nel_1, nel_2, q+1, p+1)
  jM = Array{Int64}(undef, nel_1, nel_2, q+1, p+1)
  fill!(iM, 0); fill!(jM, 0)
  for Q=1:nel_1, P=1:nel_2, pᵢ=1:p+1, qᵢ=1:q+1
    iM[Q,P,qᵢ,pᵢ] = elem_1[Q,qᵢ]
    jM[Q,P,qᵢ,pᵢ] = elem_2[P,pᵢ]
  end
  iM, jM
end


function _get_assembler_vector(elem::Matrix{Int64}, p::Int64)
  nel = size(elem,1)
  iV = Array{Int64}(undef, nel, p+1)
  fill!(iV,0)
  for t=1:nel, ti=1:p+1
    iV[t,ti] = elem[t,ti]
  end
  iV
end
