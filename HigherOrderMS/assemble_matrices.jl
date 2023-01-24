########################################################################
# Function to assemble the various matrix vector systems based on the
# connectivity of the elements
########################################################################
"""
Function to assemble the standard Xₕ × Xₕ matrices
  (u,v) = ∫ₖ u*v dx
  a(u,v) = ∫ₖ ∇(u)⋅∇(v) dx
where Xₕ = H¹Conforming
"""
function assemble_matrix(U::T, assem::MatrixAssembler, A::Function, M::Function; qorder=10) where T<:H¹Conforming
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.p
  qs,ws = gausslegendre(qorder)
  i,j,sKe = assem.iM, assem.jM, assem.vM
  sMe = similar(sKe)
  fill!(sKe,0.0); fill!(sMe,0.0)
  nel = size(els,1)
  # Initialize the element-wise local matrices
  Me = Array{Float64}(undef, p+1, p+1)
  Ke = Array{Float64}(undef, p+1, p+1)
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    hlocal = cs[2] - cs[1]
    fill!(Me, 0.0)
    fill!(Ke, 0.0)
    for k=1:lastindex(qs)
      x = (cs[2]+cs[1])*0.5 .+ 0.5*hlocal*qs[k]
      ϕᵢ = U.basis(qs[k])
      ∇ϕᵢ = ∇(U.basis, qs[k])*(2/hlocal)
      _local_matrix!(Me, (ϕᵢ,ϕᵢ), M(x), ws[k], hlocal, (p,p))
      _local_matrix!(Ke, (∇ϕᵢ,∇ϕᵢ), A(x), ws[k], hlocal, (p,p))
    end    
    for ti=1:p+1, tj=1:p+1
      sMe[t,ti,tj] = Me[ti,tj]
      sKe[t,ti,tj] = Ke[ti,tj]
    end
  end
  K = sparse(vec(i), vec(j), vec(sKe))
  M = sparse(vec(i), vec(j), vec(sMe))
  droptol!(M,1e-20), droptol!(K,1e-20)
end

"""
Function to assemble the standard Xₕ × 1 load vector
  (f,v) = ∫ₖ f*v dx
where f is a known function and Xₕ = Union{H¹Conforming, L²Conforming}
"""
function assemble_vector(U::T, assem::VectorAssembler, f::Function; qorder=10) where {T<:Union{H¹Conforming, L²Conforming}}
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.p
  qs,ws = gausslegendre(qorder)
  k, sFe = assem.iV, assem.vV
  fill!(sFe,0.0)
  nel = size(els,1)
  # Initialize the local vector
  Fe = Vector{Float64}(undef, p+1)
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    hlocal = cs[2]-cs[1]
    fill!(Fe,0.0)
    for k=1:lastindex(qs)
      x = (cs[2]+cs[1])*0.5 .+ 0.5*hlocal*qs[k]
      ϕᵢ = U.basis(-(cs[2]+cs[1])/(cs[2]-cs[1]) + 2/(cs[2]-cs[1])*x)
      _local_vector!(Fe, ϕᵢ, f(x), ws[k], hlocal, p)
    end
    for ti=1:p+1
      sFe[t,ti] = Fe[ti]
    end
  end
  F = collect(sparsevec(vec(k),vec(sFe)))
  F
end
"""
Function to assemble the Xₕ × Vₕ matrix 
    (u,Λ) = ∫ₖ u*Λ dx
Here u ∈ Xₕ=H¹Conforming, Λ ∈ Vₕ=L²Conforming
"""
function assemble_matrix(U::T1, V::T2, assem::MatrixAssembler, A::Function; qorder=10) where {T1<:H¹Conforming, T2<:L²Conforming}
  # Get the data
  trian₁ = get_trian(U) # Fine
  trian₂ = get_trian(V) # Coarse
  els₁ = trian₁.elems
  els₂ = trian₂.elems
  q = U.p
  p = V.p
  qs,ws = gausslegendre(qorder)
  i,j,sMe = assem.iM, assem.jM, assem.vM
  fill!(sMe,0.0)
  nel₁ = size(els₁,1)
  nel₂ = size(els₂,1)
  Me = Matrix{Float64}(undef, q+1, p+1)
  # The Legendre basis function Λₖⱼ with supp(Λₖⱼ) = K
  function Bₖ(x,nds)
    a,b = nds
    x̂ = -(b+a)/(b-a) + 2/(b-a)*x
    (a ≤ x ≤ b) ? V.basis(x̂) : zeros(Float64,p+1)
  end
  # Do the assembly
  for Q=1:nel₁
    CQ = trian₁.nds[trian₁.elems[Q,:]] # Fine space
    hlocal = CQ[2]-CQ[1]
    for P=1:nel₂
      CP = trian₂.nds[trian₂.elems[P,:]] # Coarse space
      Λₖ(y) = Bₖ(y,CP)
      ϕᵢ(y) = U.basis(-(CQ[2]+CQ[1])/(CQ[2]-CQ[1]) + 2/(CQ[2]-CQ[1])*y)
      fill!(Me,0.0)
      for k=1:lastindex(qs)
        x = (CQ[2]+CQ[1])*0.5 .+ 0.5*hlocal*qs[k]
        ϕᵢ = U.basis(-(CQ[2]+CQ[1])/(CQ[2]-CQ[1]) + 2/(CQ[2]-CQ[1])*x)
        Λᵢ = Λₖ(x)
        _local_matrix!(Me, (ϕᵢ,Λᵢ), A(x), ws[k], hlocal, (q,p))
      end 
      for qᵢ=1:q+1, pᵢ=1:p+1
        sMe[Q,P,qᵢ,pᵢ] = Me[qᵢ,pᵢ]
      end
    end
  end
  K = sparse(vec(i), vec(j), vec(sMe))
  droptol!(K,1e-20)
end

######### ############ ############ ############ ############ ########### ############
######### Functions to assemble the matrices containing multiscale bases  ############
######### ############ ############ ############ ############ ########### ############
"""
Function to assemble the standard Xₕ × Xₕ matrices
  (u,v) = ∫ₖ u*v dx
  a(u,v) = ∫ₖ ∇(u)⋅∇(v) dx
where Xₕ = Multiscale
"""
function assemble_matrix(U::T, assem::MatrixAssembler, A::Function, M::Function; qorder=10, num_neighbours=2) where T<:MultiScale
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.bgSpace.p
  l = U.l  
  new_els = _new_elem_matrices(els, p, l, MultiScaleSpace())
  qs,ws = gausslegendre(qorder)
  i,j,sKe = assem.iM, assem.jM, assem. vM
  sMe = similar(sKe)
  fill!(sKe,0.0); fill!(sMe,0.0)
  nel = size(els,1)
  # Initialize the element-wise local matrices
  ndofs = ((p+1)*(2l+1) < (p+1)*nel) ? (p+1)*(2l+1) : (p+1)*nel
  Me = Array{Float64}(undef,ndofs,ndofs)
  Ke = Array{Float64}(undef,ndofs,ndofs)
  vecBasis = U.basis # Arrange element-wise functions into a vector
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    b_inds = new_els[t,:]
    hlocal = cs[2]-cs[1]
    fill!(Me,0.0); fill!(Ke,0.0)
    for k=1:lastindex(qs)
      x = (cs[2]+cs[1])*0.5 .+ 0.5*hlocal*qs[k]
      ϕᵢ = [Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U; num_neighbours=num_neighbours) for i in b_inds]
      ∇ϕᵢ = [∇Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U; num_neighbours=num_neighbours) for i in b_inds]
      _local_matrix!(Ke, (∇ϕᵢ,∇ϕᵢ), A(x), ws[k], hlocal, (ndofs-1,ndofs-1))
      _local_matrix!(Me, (ϕᵢ,ϕᵢ), M(x), ws[k], hlocal, (ndofs-1,ndofs-1))
    end
    for tj=1:ndofs, ti=1:ndofs
      sMe[t,ti,tj] = Me[ti,tj]
      sKe[t,ti,tj] = Ke[ti,tj] 
    end   
  end
  K = sparse(vec(i),vec(j),vec(sKe))
  M = sparse(vec(i),vec(j),vec(sMe))
  droptol!(M,1e-20), droptol!(K,1e-20)
end 

"""
Function to assemble the standard Xₕ × 1 load vector
  (f,v) = ∫ₖ f*v dx
where f is a known function and Xₕ = MultiScale
"""
function assemble_vector(U::T, assem::VectorAssembler, f::Function; qorder=10,num_neighbours=2) where T<:MultiScale
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.bgSpace.p
  l = U.l
  new_els = _new_elem_matrices(els, p, l, MultiScaleSpace())
  qs,ws = gausslegendre(qorder)
  k,sFe = assem.iV, assem.vV
  fill!(sFe,0.0)
  nel = size(els,1)
  # Initializa the elemtn-wise vector
  ndofs = ((p+1)*(2l+1) < (p+1)*nel) ? (p+1)*(2l+1) : (p+1)*nel
  Fe = Vector{Float64}(undef,ndofs)
  fill!(Fe,0.0)
  vecBasis = vec(U.basis) # Arrange element-wise functions into a vector
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    b_inds = new_els[t,:]
    hlocal = cs[2]-cs[1]
    fill!(Fe,0.0)
    for k=1:lastindex(qs)
      x = (cs[2]+cs[1])*0.5 .+ 0.5*hlocal*qs[k]
      ϕᵢ = [Λ̃ˡₚ(x, vecBasis[i], vecBasis[i].U; num_neighbours=num_neighbours) for i in b_inds]
      _local_vector!(Fe, ϕᵢ, f(x), ws[k], hlocal, ndofs-1)
    end
    for ti=1:ndofs
      sFe[t,ti] = Fe[ti]
    end 
  end 
  F = collect(sparsevec(vec(k),vec(sFe)))
end
