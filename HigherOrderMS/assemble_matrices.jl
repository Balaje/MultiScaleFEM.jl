########################################################################
# Function to assemble the various matrix vector systems based on the
# connectivity of the elements
########################################################################
"""
Function to get the assembler for any polynomial of order p:
    (1) Stores the indices of the global matrix corresponding to the local matrices
    (2) This is useful for generating the global matrix where the local system changes w.r.t x
"""
function get_assembler(elem, p)
  nel = size(elem,1)
  new_elem, _ = new_connectivity_matrices(elem, (p,p))
  iM = Array{Int64}(undef, nel, p+1, p+1)
  jM = Array{Int64}(undef, nel, p+1, p+1)
  iV = Array{Int64}(undef, nel, p+1)
  fill!(iM,0); fill!(jM, 0); fill!(iV,0)
  for t=1:nel
    for ti=1:p+1
      iV[t,ti] = new_elem[t,ti]
      for tj=1:p+1
        iM[t,ti,tj] = new_elem[t,ti]
        jM[t,ti,tj] = new_elem[t,tj]
      end
    end
  end
  iM, jM, iV
end

function get_assembler(elem::Tuple, fespace::Tuple)
  q,p = fespace
  elem_1, elem_2 = elem
  nel_1 = size(elem_1,1)
  nel_2 = size(elem_2,1)
  new_elem_1 = new_connectivity_matrices(elem_1,(q,q))[1]
  new_elem_2 = new_connectivity_matrices(elem_2,(p,p))[2]
  iM = Array{Int64}(undef, nel_1, nel_2, q+1, p+1)
  jM = Array{Int64}(undef, nel_1, nel_2, q+1, p+1)
  iV = Array{Int64}(undef, nel_1, nel_2, p+1)
  fill!(iM, 0); fill!(iV, 0); fill!(jM, 0)
  for Q=1:nel_1
    for P=1:nel_2
      for pᵢ=1:p+1
        iV[Q,P,pᵢ] = new_elem_2[P,pᵢ]
        for qᵢ=1:q+1
          iM[Q,P,qᵢ,pᵢ] = new_elem_2[P,pᵢ]
          jM[Q,P,qᵢ,pᵢ] = new_elem_1[Q,qᵢ]
        end
      end
    end
  end
  iM, jM, iV
end
"""
Function to assemble the standard H¹(D) × H¹(D) matrix and vector:
    (u,v) = ∫ₖ u*v dx
    a(u,v) = ∫ₖ ∇(u)⋅∇(v) dx
    (f,v)  = ∫ₖ f*v dx
Returns the tuple containing (mass, stiffness, load) vectors
"""
function assemble_matrix_H¹_H¹(ijk, nodes, els, A::Function, f::Function, p; qorder=10)
  i,j,k = ijk
  nel = size(i,1)
  hlocal = nodes[2]-nodes[1]
  quad = gausslegendre(qorder)
  sKe = Array{Float64}(undef, nel, p+1, p+1)
  sMe = Array{Float64}(undef, nel, p+1, p+1)
  sFe = Array{Float64}(undef, nel, p+1)
  fill!(sKe,0.0); fill!(sMe,0.0); fill!(sFe,0.0)
  # Do the assembly
  for t=1:nel
    cs = nodes[els[t,:],:]
    Me, Ke, Fe = _local_matrix_vector_H¹_H¹(cs, A, f, quad, hlocal, p)
    for ti=1:p+1
      sFe[t,ti] = Fe[ti]
      for tj=1:p+1
        sMe[t,ti,tj] = Me[ti,tj]
        sKe[t,ti,tj] = Ke[ti,tj]
      end
    end
  end
  K = sparse(vec(i), vec(j), vec(sKe))
  M = sparse(vec(i), vec(j), vec(sMe))
  F = collect(sparsevec(vec(k), vec(sFe)))
  droptol!(M,1e-20), droptol!(K,1e-20), F
end

"""
Function to assemble the H¹(D) × Vₕᵖ(K) matrix and the 1 × Vₕᵖ(K) vector
    (u,Λₖ) = ∫ₖ u*Λₖ dx: Matrix
    (f,Λₖ) = ∫ₖ f*Λₖ dx: Vector
Here u ∈ H¹₀(K), Λₖ ∈ Vₕᵖ(K) and f is a known function
"""
function assemble_matrix_vector_H¹_Vₕᵖ(ijk, nodes::Tuple, elems::Tuple,
                                       f::Function, fespace; qorder=10)
  # Get the data
  iM, jM, iF = ijk
  nodes_coarse, nodes_fine = nodes
  elem_coarse, elem_fine = elems
  q,p = fespace
  nel_coarse = size(elem_coarse,1)
  nel_fine = size(elem_fine,1)
  quad = gausslegendre(qorder)

  sMe = Array{Float64}(undef, nel_fine, nel_coarse, q+1, p+1)
  sFe = Array{Float64}(undef, nel_fine, nel_coarse, p+1)
  fill!(sMe, 0.0);  fill!(sFe, 0.0)

  # The Legendre basis function Λₖⱼ with supp(Λₖj) = K
  function Bₖ(x,p,nds)
    a,b=nds
    x̂ = -(a+b)/(b-a) + 2/(b-a)*x
    (a ≤ x ≤ b) ? ψ̂(x̂,p) : zeros(Float64,p+1)
    # ψ̂(x̂,p)
  end

  # Do the assembly
  for Q=1:nel_fine
    CQ = nodes_fine[elem_fine[Q,:]]
    for qᵢ=1:q+1
      for P=1:nel_coarse
        CP = nodes_coarse[elem_coarse[P,:]]
        Λₖ(y) = Bₖ(y,p,CP)
        Mlocal = _local_matrix_H¹_Vₕᵖ(Λₖ, (CQ[1],CQ[2]), fespace; h=CQ[2]-CQ[1], qorder=qorder)
        Flocal = _local_vector_Vₕᵖ(f, Λₖ, (CQ[1],CQ[2]), fespace; h=CQ[2]-CQ[1], qorder=qorder)
        for pᵢ=1:p+1
          sMe[Q,P,qᵢ,pᵢ] = Mlocal[qᵢ,pᵢ]
          sFe[Q,P,pᵢ] = Flocal[pᵢ]
        end
      end
    end
  end
  K = sparse(vec(jM), vec(iM), vec(sMe))
  F = collect(sparsevec(vec(iF), vec(sFe)))
  K,F
end

########################################################################################################################################
"""
Function to assemble the Multiscale matrix-vector system
"""
function assemble_matrix_MS_MS(nodes, els, ms_elems, RˡₕΛₖ::Matrix{Rˡₕ}, A::Function, f::Function, fespace; qorder=10, Nfine=100, plot_basis=1)
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
  droptol!(K, 1e-20), droptol!(M, 1e-20), F
end
########################################################################################################################################
