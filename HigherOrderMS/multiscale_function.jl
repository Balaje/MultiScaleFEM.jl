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
function uᵐˢₕ(x, sol, R::Matrix{Rˡₕ}, nodes, elem::Matrix{Int64}, fespace)
    _,p = fespace
    new_elem, _ = new_connectivity_matrices(elem, (p,p))  
    nel = size(elem,1)    
    for i=1:nel
      nds_elem = nodes[elem[i,:]]    
      if(nds_elem[1] ≤ x ≤ nds_elem[2])            
        val=0
        uh_elem = sol[new_elem[i,:]]
        for j=1:p+1
          val = val + Λ̃ₖˡ(x, R[i,j]; fespace=fespace)*uh_elem[j]
        end 
        return val
      end 
    end         
end 