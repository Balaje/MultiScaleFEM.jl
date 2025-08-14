using Pkg
# Pkg.add("FastGaussQuadrature")
# Pkg.add("DoubleFloats")
# Pkg.add("LaTeXStrings")
# Pkg.add("ColorSchemes")
# Pkg.add("Gridap")
# Pkg.add("DoubleFloats")
# Pkg.add("Plots")
# Pkg.add("Quadmath")
# Pkg.add("DoubleFloats")

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays   
using Gridap
using DoubleFloats
using ProgressMeter

# using LaTeXStrings
# using ColorSchemes


include("coarse_to_fine.jl")
include("basis-functions.jl")
include("assemble_matrices.jl")
include("multiscale_basis-functions.jl")
include("time-dependent.jl")
include("corrected_basis.jl");


## Define the \ operator for higher precision dataTypes
using Quadmath
using DoubleFloats

import LinearAlgebra.\

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
