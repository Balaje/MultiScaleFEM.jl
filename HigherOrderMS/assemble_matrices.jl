########################################################################
# Function to assemble the various matrix vector systems based on the
# connectivity of the elements
########################################################################
"""
Function to assemble the standard H¹(D) × H¹(D) matrix and vector:
    (u,v) = ∫ₖ u*v dx
    a(u,v) = ∫ₖ ∇(u)⋅∇(v) dx
    (f,v)  = ∫ₖ f*v dx
Returns the tuple containing (mass, stiffness, load) vectors
"""
function assemble_matrix(U::T, assem::MatrixAssembler, A::Function; qorder=10) where {T<:Union{H¹Conforming,MultiScale}}
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.p
  quad = gausslegendre(qorder)
  i,j = assem.iM, assem.jM
  nel = size(els,1)
  # Initialize the element-wise local matrices
  sKe = Array{Float64}(undef, size(i))
  sMe = Array{Float64}(undef, size(i))
  fill!(sKe,0.0); fill!(sMe,0.0)
  Me = Array{Float64}(undef, p+1, p+1)
  Ke = Array{Float64}(undef, p+1, p+1)
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    hlocal = cs[2] - cs[1]
    _local_matrix!(Me, cs, (U.basis,U.basis), A, quad, hlocal, (p,p))
    _local_matrix!(Ke, cs, (y->∇(U.basis,y),y->∇(U.basis,y)), A, quad, hlocal, (p,p))
    for ti=1:p+1, tj=1:p+1
      sMe[t,ti,tj] = Me[ti,tj]
      sKe[t,ti,tj] = Ke[ti,tj]
    end
  end
  K = sparse(vec(i), vec(j), vec(sKe))
  M = sparse(vec(i), vec(j), vec(sMe))
  droptol!(M,1e-20), droptol!(K,1e-20)
end

function assemble_vector(U::T, assem::VectorAssembler, f::Function; qorder=10) where {T<:FiniteElementSpace}
  trian = get_trian(U)
  nodes = trian.nds
  els = trian.elems
  p = U.p
  quad = gausslegendre(qorder)
  k = assem.iV
  nel = size(els,1)
  # Initialize the local vector
  sFe = Array{Float64}(undef, size(k))
  fill!(sFe,0.0)
  Fe = Vector{Float64}(undef, p+1)
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    hlocal = cs[2]-cs[1]
    _local_vector!(Fe, cs, U.basis, f, quad, hlocal, p)
    for ti=1:p+1
      sFe[t,ti] = Fe[ti]
    end
  end
  F = collect(sparsevec(vec(k),vec(sFe)))
end
"""
Function to assemble the H¹(D) × Vₕᵖ(K) matrix and the 1 × Vₕᵖ(K) vector
    (u,Λₖ) = ∫ₖ u*Λₖ dx: Matrix
    (f,Λₖ) = ∫ₖ f*Λₖ dx: Vector
Here u ∈ H¹₀(K), Λₖ ∈ Vₕᵖ(K) and f is a known function
"""
function assemble_matrix(U::T1, V::T2, assem::MatrixAssembler, A::Function; qorder=10) where {T1<:H¹Conforming, T2<:L²Conforming}  
  # Get the data
  trian₁ = get_trian(U) # Fine
  trian₂ = get_trian(V) # Coarse
  els₁ = U.elem
  els₂ = V.elem
  q = U.p
  p = V.p
  quad = gausslegendre(qorder)
  i,j = assem.iM, assem.jM
  nel₁ = size(els₁,1)
  nel₂ = size(els₂,1)
  sMe = Array{Float64}(undef, size(i))
  fill!(sMe, 0.0);
  Me = Matrix{Float64}(undef, q+1, p+1)
  # The Legendre basis function Λₖⱼ with supp(Λₖⱼ) = K
  function Bₖ(x,nds)
    a,b=nds
    x̂ = -(a+b)/(b-a) + 2/(b-a)*x
    (a ≤ x ≤ b) ? V.basis(x̂) : zeros(Float64,p+1)
  end
  # Do the assembly
  for Q=1:nel₁
    CQ = trian₁.nds[trian₁.elems[Q,:]]
    hlocal = CQ[2]-CQ[1]
    for qᵢ=1:q+1, P=1:nel₂
      CP = trian₂.nds[trian₂.elems[P,:]]
      Λₖ(y) = Bₖ(y,CP)
      _local_matrix!(Me, CQ, (U.basis,Λₖ), A, quad, hlocal, (q,p))
      for pᵢ=1:p+1
        sMe[Q,P,qᵢ,pᵢ] = Me[qᵢ,pᵢ]
      end
    end
  end
  K = sparse(vec(i), vec(j), vec(sMe))
  dropzeros!(K)
end

########################################################################################################################################
"""
Function to assemble the Multiscale matrix-vector system
"""
function assemble_matrix_MS_MS(nodes, els, ms_elems, RˡₕΛₖ::Matrix{Rˡₕ},
                               A::Function, f::Function, fespace;
                               qorder=10, plot_basis=-1)
  q,p = fespace
  nel = size(els,1)
  hlocal = nodes[2]-nodes[1]
  quad  = gausslegendre(qorder)
  sKe = Array{Float64}(undef, nel, (2l+1)*(p+1), (2l+1)*(p+1))
  sMe = Array{Float64}(undef, nel, (2l+1)*(p+1), (2l+1)*(p+1))
  sFe = Array{Float64}(undef, nel, (2l+1)*(p+1))
  fill!(sKe,0.0); fill!(sMe,0.0); fill!(sFe,0.0)
  if(plot_basis > 0)
    plt = plot()
    for i=1:p+1
      xs = RˡₕΛₖ[plot_basis,i].nds
      fxs = map(x->∇Λ̃ₖˡ(x, RˡₕΛₖ[plot_basis,i]; fespace=fespace), xs)
      plot!(plt, xs, fxs, lw=2, label="Basis "*string(i))
      xlims!(plt, (0,1))
    end
    savefig(plt, "local_basis.pdf")
  end
  # Do the assembly
  K = spzeros(Float64, maximum(ms_elems), maximum(ms_elems))
  M = spzeros(Float64, maximum(ms_elems), maximum(ms_elems))
  F = Vector{Float64}(undef, maximum(ms_elems))
  # Let us use the naive way to assemble the matrices for now
  for t=1:nel
    cs = nodes[els[t,:],:]
    el = nonzeros(sparsevec(ms_elems[t,:]))
    start = (t-l)>0 ? t-l : 1
    last = (t+l)<nel ? t+l : nel
    R = RˡₕΛₖ[start:last,:]
    R = permutedims(R,[2,1])
    Ke,Me,Fe = _local_matrix_vector_MS_MS(cs, A, f, quad, hlocal, fespace,
                                          length(el), vec(R))
    for i=1:lastindex(el)
      F[el[i]] += Fe[i]
      for j=1:lastindex(el)
        M[el[i],el[j]] += Me[i,j]
        K[el[i],el[j]] += Ke[i,j]
      end
    end
  end
  droptol!(K, 1e-20), droptol!(M, 1e-20), collect(droptol!(sparsevec(F),1e-20))
end
########################################################################################################################################
