################################################################################################
# Function to evaluate the multiscale basis at any point x
################################################################################################

"""
Function to evaluate the multiscale basis at any point x:
    (1) Requires the multiscale-basis vectors (R)
    (2) MSFEM solution (sol)
    (3) Nodes of the mesh (nodes)
    (4) Connectivity of the mesh
    (5) Order of polynomial (p)
"""
function uᵐˢₕ(x, sol, R::Matrix{Rˡₕ}, nodes, elem::Matrix{Int64}, ms_elem::Matrix{Int64}, fespace)
  q,p = fespace
  nel = size(elem,1)
  for i=1:nel
    nds_elem = nodes[elem[i,:]]
    if(nds_elem[1] ≤ x ≤ nds_elem[2])
      val=0
      uh_elem = sol[ms_elem[i,:]]
      basis_elem = R[ms_elem[i,:],i]
      for j=1:length(uh_elem)
        val = val + Λ̃ₖˡ(x, R[i,j], fespace)*uh_elem[j]
      end
      return val
    end
  end
end
