########################################################################
# Function to assemble the matrix vector system for the MS method
########################################################################
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
        xs = getindex.(RˡₕΛₖ[plot_basis,i].Ω.grid.node_coords,1)        
        fxs = map(x->Λ̃ₖˡ(x, RˡₕΛₖ[plot_basis,i]), xs)     
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
