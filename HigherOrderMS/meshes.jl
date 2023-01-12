#####################################################################
# File containing all the data structures for the mesh connectivity #
#####################################################################

struct MeshType <: Any end
"""
mutable struct 𝒯 <: MeshType
  H::Float64
  nds::AbstractVector{Float64}
  elems::Matrix{Int64}
  elemsₗ::Matrix{Int64}
end

Structure to store the information of the coarse mesh
"""
mutable struct 𝒯 <: MeshType
  H::Float64
  nds::AbstractVector{Float64}
  elems::Matrix{Int64}
  pdegree::Int64
  l::Int64
  elemsₗ::Matrix{Int64}
end
function 𝒯(domain::Tuple, N::Int64, p::Int64; l=0)
  @assert l isa Int
  H = (domain[2]-domain[1])/N
  nodes = domain[1]:H:domain[2]
  elems = Matrix{Int64}(undef,N,p+1)
  elemsₗ = Matrix{Int64}(undef,N,(p+1)*(2l+1))
  fill!(elems,0); fill!(elemsₗ,0)
  for i=1:N, j=0:p
      elems[i,j+1] = i+j + (p-1)*(i-1)
  end
  for i=1:N
    start = ((i-l) ≤ 0) ? 1 : (i-l)
    last = ((i+l) ≥ N) ? N : (i+l)
    elinds = vec(permutedims(elems[start:last,:],[2,1]))
    elemsₗ[i,1:length(elinds)] = elinds
  end
  𝒯(H, nodes, elems, p, l, elemsₗ)
end

"""
mutable struct 𝒯ₕ <: Any
  h::Float64
  nds::AbstractVector{Float64}
  elems::Matrix{Int64}
  pdegree::Int64
end

Structure to store the information about the fine mesh.
"""
mutable struct 𝒯ₕ <: MeshType
  h::Float64
  nds::AbstractVector{Float64}
  elems::Matrix{Int64}
  pdegree::Int64
end
function 𝒯ₕ(domain::Tuple, N::Int64, p::Int64)
  @assert l isa Int
  h = (domain[2]-domain[1])/N
  nodes = domain[1]:h:domain[2]
  elems = Matrix{Int64}(undef,N,p+1)
  fill!(elems,0)
  for i=1:N
    for j=0:p
      elems[i,j+1] = i+j + (p-1)*(i-1)
    end
  end
  𝒯ₕ(h, nodes, elems, p)
end
