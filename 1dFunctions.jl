###############################################################
# Contains functions to implement the essential steps in 1D FEM
# DO NOT MODIFY.
###############################################################
"""
Function to generate the connectivity information
"""
function mesh(domain::Tuple, N::Int64)
  h = (domain[2]-domain[1])/N
  nodes = domain[1]:h:domain[2]
  elem = zeros(Int64,N,2)
  for i in 1:N, j in 0:1
    elem[i,j+1] = i+j
  end

  nodes, elem
end


"""
Function to generate the basis functions φ̂(x̂) on the reference element (-1,1)
(Standard type)
"""
φ̂(x) = [0.5 - 0.5*x[1], 0.5 + 0.5*x[1]]
function φ̂!(res, x)
  res[1] = 0.5 - 0.5*x[1]
  res[2] = 0.5 + 0.5*x[1]
end
"""
Function to generate the gradient of basis functions ∇̂φ̂(x̂) on the reference element (-1,1)
(Standard type)
"""
∇φ̂(x) = [-0.5, 0.5]
function ∇φ̂!(res, x)
  res[1] = -0.5
  res[2] = 0.5
end

"""
Function to generate the local matrix-vector system using the given data
"""
function local_matrix_vector!(cache, xn, c::Function, f::Function, quad)
  Me, Ke, Fe, res, res1, Nothing = cache
  fill!(Me, 0.); fill!(Ke, 0.); fill!(Fe, 0.);
  fill!(res, 0.); fill!(res1, 0.)

  qs,ws = quad
  J = (xn[2] - xn[1])/2

  for q=1:length(qs)
    x̂ = qs[q]
    x = (xn[2] + xn[1])/2 .+ (xn[2] - xn[1])/2*x̂
    φ̂!(res, x̂); ∇φ̂!(res1, x̂)
    for i=1:2
      ϕᵢ = res[i]; ∇ϕᵢ = res1[i]
      Fe[i] += ws[q]*( f(x)*ϕᵢ )*J
      for j=1:2
        ϕⱼ = res[j]; ∇ϕⱼ = res1[j]
        Me[i,j] += ws[q]*( ϕᵢ * ϕⱼ )*J
        Ke[i,j] += ws[q]*( c(x).^2 * ∇ϕᵢ * ∇ϕⱼ )*J^-1
      end
    end
  end

end

"""
Function to get the assembler
"""
function get_assembler(elem)
  nel = size(elem,1)
  iM = Array{Int64}(undef, nel, 2, 2)
  jM = Array{Int64}(undef, nel, 2, 2)
  iV = Array{Int64}(undef, nel, 2)
  for t=1:nel
    for ti=1:2
      iV[t,ti] = elem[t,ti]
      for tj=1:2
        iM[t,ti,tj] = elem[t,ti]
        jM[t,ti,tj] = elem[t,tj]
      end
    end
  end
  iM, jM, iV
end

"""
Function to assemble the local matrices
"""
function assemble_matrix(ij, nodes, c::Function; quad=gausslegendre(2),
  local_matrix_vector_function=local_matrix_vector!, N=100)
  i,j = ij
  nel = size(i,1)
  # Matrices
  sKe = Array{Float64}(undef, nel, 2, 2)
  sMe = Array{Float64}(undef, nel, 2, 2)

  # Preallocate local matrices
  Me = Array{Float64}(undef,2,2)
  Ke = Array{Float64}(undef, 2,2)
  Fe = Array{Float64}(undef,2)
  res = Vector{Float64}(undef,2)
  res1 = Vector{Float64}(undef,2)

  cache = Me, Ke, Fe, res, res1, N
  # Do the assembly
  for t=1:nel
    cs = view(nodes, view(i, t,:,1))
    local_matrix_vector_function(cache, cs, c, x->1, quad)
    for ti=1:2, tj=1:2
      sMe[t,ti,tj] = Me[ti,tj]
      sKe[t,ti,tj] = Ke[ti,tj]
    end
  end
  # Return the sparse version
  K = sparse(vec(i), vec(j), vec(sKe))
  M = sparse(vec(i), vec(j), vec(sMe))
  M, K
end

"""
Function to assemble the local vector
"""
function assemble_vector(i, nodes, f::Function; quad=gausslegendre(2), 
  local_matrix_vector_function=local_matrix_vector!, N=100)
  nel = size(i,1)

  # Vector
  sFe = Array{Float64}(undef, nel, 2)

  # Preallocate local matrices
  Me = Array{Float64}(undef,2,2)
  Ke = Array{Float64}(undef,2,2)
  Fe = Vector{Float64}(undef,2)
  res = Vector{Float64}(undef,2)
  res1 = Vector{Float64}(undef,2)

  cache = Me, Ke, Fe, res, res1, N

  for t=1:nel
    cs = view(nodes, view(i, t,:))
    local_matrix_vector_function(cache, cs, x->1, f, quad)
    for ti=1:2
      sFe[t,ti] = Fe[ti]
    end
  end
  collect(sparsevec(vec(i), vec(sFe)))
end

"""
Function to compute the L² error
"""
function l2err(uh::AbstractVector, u::Function, nodes, elem; quad=gausslegendre(6))
  qs,ws = quad
  J = (nodes[2]-nodes[1])*0.5
  uh_elem = uh[elem]
  u_elem = u.(nodes[elem])
  nel = size(elem,1)
  err = 0
  res = zeros(Float64,2)
  for t=1:nel
    for q=1:length(qs)
      x̂ = qs[q]
      res = φ̂(x̂)
      uhx = uh_elem[t,1]*res[1] + uh_elem[t,2]*res[2]
      ux = u_elem[t,1]*res[1] + u_elem[t,2]*res[2]
      err += ws[q]*(ux - uhx)^2*J
    end
  end
  sqrt(err)
end
