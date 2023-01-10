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
Function to assemble the H¹(D) × L²(D) matrix:
    (u,v) = ∫ₖ u*v dx
Here u ∈ H¹₀(K), v ∈ L²(K). Since the local matrices do not change w.r.t the elements, we can assemble the elements directly without looping over elements.
"""
function assemble_matrix_H¹_L²(node, elem, (p₁,p₂))
    hlocal = node[2,1]-node[1,1]
    Mlocal = _localmassmatrix_H¹_L²((p₁,p₂); h=hlocal) #(p+1) × p matrix
    nel = size(elem,1)    
    new_elem, new_elem_1 = new_connectivity_matrices(elem, (p₁,p₂))
    # Size of the index vectors = (dim*p+1)*(dim*p)*nel
    ii = Vector{Int64}(undef, (p₁+1)*(p₂+1)*nel)
    jj = Vector{Int64}(undef, (p₁+1)*(p₂+1)*nel)
    sA = Vector{Float64}(undef, (p₁+1)*(p₂+1)*nel)
    fill!(ii,0)
    fill!(jj,0)
    fill!(sA,0.0)
    index = 0
    for i=1:p₁+1, j=1:p₂+1
      ii[index+1:index+nel] = new_elem[:,i]
      jj[index+1:index+nel] = new_elem_1[:,j]
      sA[index+1:index+nel] .= Mlocal[i,j]
      index = index+nel
    end
    droptol!(sparse(ii, jj, sA), 1e-20)
end
"""
Function to assemble the L²(D) × L²(D) matrix
    (u,v) = ∫ₖ u*v dx
Here u, v ∈ L²(K)
"""
function assemble_matrix_L²_L²(node, elem, p)
    hlocal = node[2,1]-node[1,1]
    Mlocal = _localmassmatrix_L²_L²(p; h=hlocal)
    nel = size(elem,1)
    _, new_elem = new_connectivity_matrices(elem, (p,p))
    ii = Vector{Int64}(undef, (p+1)*(p+1)*nel)
    jj = Vector{Int64}(undef, (p+1)*(p+1)*nel)
    sA = Vector{Float64}(undef, (p+1)*(p+1)*nel)
    fill!(ii,0)
    fill!(jj,0)
    fill!(sA,0.0)
    index = 0
    for i=1:p+1, j=1:p+1
      @inbounds begin
        ii[index+1:index+nel] = new_elem[:,i]
        jj[index+1:index+nel] = new_elem[:,j]
        sA[index+1:index+nel] .= Mlocal[i,j]
        index = index+nel
      end
    end
    droptol!(sparse(ii, jj, sA), 1e-20)
end  

"""
Function to assemble the L²(D) vector
"""
function assemble_vector_L²(node, elem, p, f::Function; qorder=10)
    hlocal = node[2,1] - node[1,1]
    nel = size(elem,1)
    res = Vector{Float64}(undef, nel*(p+1))
    fill!(res,0.)    
    _, new_elem = new_connectivity_matrices(elem, (p,p))
    for i=1:nel
      cs = node[elem[i,:],:]
      Felocal = _localvector_L²(cs, p, f; qorder=qorder, h=hlocal)
      res[new_elem[i,1]:new_elem[i,end]] = Felocal
    end
    res
    res[abs.(res) .< 1e-20] .= 0
    res
end

########################################################################################################################################
########################################################################################################################################
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
  ########################################################################################################################################
  ########################################################################################################################################