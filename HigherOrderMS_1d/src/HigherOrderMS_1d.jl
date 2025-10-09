__precompile__()

module HigherOrderMS_1d

include("./coarse_to_fine.jl")
using HigherOrderMS_1d.CoarseToFine: coarse_space_to_fine_space
export coarse_space_to_fine_space

include("./basis-functions.jl")
using HigherOrderMS_1d.CoarseBases: Λₖ!, ιₖ
export Λₖ!, ιₖ

include("./assemble_matrices.jl")
using HigherOrderMS_1d.Assemblers: FineScaleSpace, get_saddle_point_problem, assemble_stiffness_matrix, assemble_mass_matrix, assemble_lm_matrix, assemble_load_vector
export FineScaleSpace, get_saddle_point_problem, assemble_stiffness_matrix, assemble_mass_matrix, assemble_lm_matrix, assemble_load_vector

include("./multiscale_basis-functions.jl")
using HigherOrderMS_1d.MultiscaleBases: compute_ms_basis, compute_stabilized_ms_basis
export compute_ms_basis, compute_stabilized_ms_basis

include("./corrected_basis.jl")
using HigherOrderMS_1d.AdditionalCorrections: compute_additional_correction_basis
export compute_additional_correction_basis



## Define the \ operator for higher precision dataTypes
using Quadmath
using DoubleFloats
import LinearAlgebra.\
using SparseArrays
using LinearAlgebra

function \(A::SparseMatrixCSC{T1}, b::AbstractVecOrMat{T2}) where {T1<:Double64, T2<:Real}
  LU = copy(A);
  LU = LinearAlgebra.generic_lufact!(LU)
  LU\b
end


function \(A::SparseMatrixCSC{T1}, b::AbstractVecOrMat{T2}) where {T1<:Float128, T2<:Real}
  LU = copy(A);
  LU = LinearAlgebra.generic_lufact!(LU)
  LU\b
end

end