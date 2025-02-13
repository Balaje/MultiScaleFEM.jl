module Assemblers

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData

using SparseArrays
using SplitApplyCombine
using ProgressMeter

##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #
# Contains the interfaces to build the coarse scale fespaces and the saddle point matrices
##### ###### ###### ###### ###### ###### ###### ###### ###### ####### ####### ####### ####### #

"""
Function to assemble the standard stiffness matrix given the diffusion coefficient A(x)
"""
function assemble_stima(fespace::FESpace, A, qorder::Int64)
  Ω = get_triangulation(fespace)
  dΩ = Measure(Ω, qorder)
  a(u,v) = ∫( A*∇(u) ⊙ ∇(v) )dΩ
  assemble_matrix(a, fespace, fespace)  
end

"""
Function to assemble the standard mass matrix given the reaction coefficient A(x)
"""
function assemble_massma(fespace::FESpace, A, qorder::Int64)
  Ω = get_triangulation(fespace)
  dΩ = Measure(Ω, qorder)
  a(u,v) = ∫( A*(u)⋅(v) )dΩ
  assemble_matrix(a, fespace, fespace)  
end


"""
Function to assemble the standard load vector given the load f(x)
"""
function assemble_loadvec(fespace::FESpace, f, qorder::Int64)
  Ω = get_triangulation(fespace)
  dΩ = Measure(Ω, qorder)
  l(v) = ∫( f*v )dΩ
  assemble_vector(l, fespace)
end

"""
Scaled monomial bases at each coarse rectanles. 
The L² functions in the higher order MS Method is the tensor product of the Legendre polynomials in the cell
"""
function Λₖ(x::Point, nds_x::NTuple{2,Float64}, nds_y::NTuple{2,Float64}, p::Int64, αβ::NTuple{2,Int64})  
  α,β = αβ
  Λₖ!(x[1], nds_x, p, α+1)*Λₖ!(x[2], nds_y, p, β+1)
end

function LP!(cache::Vector{Float64}, x::Float64)
  p = size(cache,1) - 1
  if(p==0)
    cache[1] = 1.0
  elseif(p==1)
    cache[1] = 1.0
    cache[2] = x
  else
    cache[1] = 1.0
    cache[2] = x
    for j=2:p
      cache[j+1] = ((2j-1)/j*x*cache[j] - (j-1)/j*cache[j-1])
    end
  end
  cache
end  
"""
Shifted Legendre Polynomial with support (a,b)
"""
function Λₖ!(x, nds::Tuple{Float64,Float64}, p::Int64, j::Int64)
  a,b = nds
  cache = Vector{Float64}(undef, p+1)
  fill!(cache,0.0)
  if(a < x <  b)
    xhat = -(b+a)/(b-a) + 2.0*x/(b-a)
    LP!(cache, xhat)
  end
  cache[j]#/sqrt((b-a)/(2j-1)/2)
end

function poly_exps(p::Int64)
  X = ones(Int64,p+1)*(0:1:p)';
  Y = (0:1:p)*ones(Int64,p+1)';
  map((a,b)->(a,b), X, Y)
end


"""
assemble_ms_matrix(B::Vector{SparseMatrix}, K::SparseMatrix, p::Int64):
1) B - Multiscale bases on the coarse scale.
2) K - Fine-scale matrix (eg. stiffness/mass)
3) p - Polynomial approximation order

Function to assemble the multiscale matrix corresponding to the fine-scale matrix K
"""
function assemble_ms_matrix(B, K)  
  B'*K*B
end
function assemble_ms_matrix(A, K, B)
  A'*K*B
end

"""
assemble_ms_loadvec(B::Vector{SparseMatrix}, F::Vector, p::Int64):
1) B - Multiscale bases on the coarse scale.
2) F - Fine-scale vector (eg. load vector)
3) p - Polynomial approximation order

Function to assemble the multiscale matrix corresponding to the fine-scale matrix K
"""
function assemble_ms_loadvec(B, F)
  B'*F
end

end