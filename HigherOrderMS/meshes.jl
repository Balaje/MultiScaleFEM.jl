#####################################################################
# File containing all the data structures for the mesh connectivity #
#####################################################################

abstract type MeshType <: Any end
"""
mutable struct 𝒯 <: MeshType
  H::Float64
  nds::AbstractVector{Float64}
  elems::Matrix{Int64}
end

Structure to store the information of the coarse mesh
"""
mutable struct 𝒯 <: MeshType
  H::Float64
  nds::AbstractVector{Float64}
  elems::Matrix{Int64}
end
function 𝒯(domain::Tuple, N::Int64)
  H = (domain[2]-domain[1])/N
  nodes = domain[1]:H:domain[2]
  elems = Matrix{Int64}(undef,N,2)
  fill!(elems,0);
  for i=1:N, j=0:1
      elems[i,j+1] = i+j
  end
  𝒯(H, nodes, elems)
end

mutable struct Nˡ <: MeshType
  H::Float64
  nds::AbstractVector{Float64}
  elems::Matrix{Int64}
end 
function Main.getindex(T::A, inds::AbstractVector{Int64}) where A<:MeshType
  H = T.H
  elems = T.elems  
  nds = T.nds
  nds_new = nds[elems[inds[1],1]:elems[inds[end],2]]
  elems_new = elems[inds,:]
  elems_new = elems_new .- minimum(elems_new) .+ 1
  Nˡ(H, nds_new, elems_new)
end 