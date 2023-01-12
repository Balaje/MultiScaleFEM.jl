######################################################################
# File containing the definition of the FESpaces used in the problem #
######################################################################
struct FiniteElementSpace <: Any end

mutable struct H¹Conforming <: FiniteElementSpace
  𝒯::MeshType
  basisFunctions::Vector{}
  assembler::Matrix{Int64}
end

mutable struct L²Conforming <: FiniteElementSpace
  𝒯::MeshType
  assembler::Matrix{Int64}
end

mutable struct MultiScale <: FiniteElementSpace
  𝒯::MeshType
end
