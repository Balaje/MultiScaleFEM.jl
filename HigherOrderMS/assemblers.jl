############################################################################
# Contains data structures to generate the assemblers of the inner-products
############################################################################
abstract type Assembler <: Any end

"""
mutable struct MatrixAssembler <: Assembler
  iM::Array{Int64}
  jM::Array{Int64}
end

Structure containing the local--global map for assembling the matrices
"""
mutable struct MatrixAssembler{T₁<:FiniteElementSpace, T₂<:FiniteElementSpace} <: Assembler
  U::T₁
  V::T₂
  iM::Array{Int64}
  jM::Array{Int64}
end
function MatrixAssembler(U::A, V::B) where {A<:FiniteElementSpace,
                                            B<:FiniteElementSpace}
  if(typeof(U) == typeof(V))
    iM, jM, _ = _get_assembler(U.elem, U.p)
    MatrixAssembler(U,V,iM,jM)
  else
    iM, jM, _ = _get_assembler((U.elem,V.elem), (U.p,V.p))
    MatrixAssembler(U,V,iM,jM)
  end
end

"""
mutable struct VectorAssembler <: Assembler
  iV::Array{Int64}
end

Structure containing the local--global map for assembling the vectors
"""
mutable struct VectorAssembler{A<:FiniteElementSpace, B<:FiniteElementSpace} <: Assembler
  U::A
  V::B
  iV::Array{Int64}
end
function VectorAssembler(U::A, V::B) where {A<:FiniteElementSpace,
                                            B<:FiniteElementSpace}
  if(typeof(U) == typeof(V))
    _, _, iV = _get_assembler(V.elem,V.p)
    VectorAssembler(V,V,iV)
  else
    _, _, iV = _get_assembler((U.elem,V.elem),(U.p,V.p))
    VectorAssembler(U,V,iV)
  end
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
function MatrixVectorAssembler(U::A, V::B) where {A<:FiniteElementSpace,
                                                  B<:FiniteElementSpace}
  mAssem = MatrixAssembler(U, V)
  vAssem = VectorAssembler(U, V)
  MatrixVectorAssembler(mAssem, vAssem)
end

function get_fespaces(assem::MatrixAssembler)
  assem.U, assem.V
end
function get_fespaces(assem::VectorAssembler)
  assem.U
end
function get_fespaces(assem::MatrixVectorAssembler)
  (assem.mAssem.U, assem.mAssem.V), assem.vAssem.U
end

#############
function _get_assembler(elem, p)
  nel = size(elem,1)
  iM = Array{Int64}(undef, nel, p+1, p+1)
  jM = Array{Int64}(undef, nel, p+1, p+1)
  iV = Array{Int64}(undef, nel, p+1)
  fill!(iM,0); fill!(jM, 0); fill!(iV,0)
  for t=1:nel
    for ti=1:p+1
      iV[t,ti] = elem[t,ti]
      for tj=1:p+1
        iM[t,ti,tj] = elem[t,ti]
        jM[t,ti,tj] = elem[t,tj]
      end
    end
  end
  iM, jM, iV
end

function _get_assembler(elem::Tuple, fespace::Tuple)
  q,p = fespace
  elem_1, elem_2 = elem
  nel_1 = size(elem_1,1)
  nel_2 = size(elem_2,1)
  iM = Array{Int64}(undef, nel_1, nel_2, q+1, p+1)
  jM = Array{Int64}(undef, nel_1, nel_2, q+1, p+1)
  iV = Array{Int64}(undef, nel_1, nel_2, p+1)
  fill!(iM, 0); fill!(iV, 0); fill!(jM, 0)
  for Q=1:nel_1
    for P=1:nel_2
      for pᵢ=1:p+1
        iV[Q,P,pᵢ] = elem_2[P,pᵢ]
        for qᵢ=1:q+1
          iM[Q,P,qᵢ,pᵢ] = elem_2[P,pᵢ]
          jM[Q,P,qᵢ,pᵢ] = elem_1[Q,qᵢ]
        end
      end
    end
  end
  jM, iM, iV
end
