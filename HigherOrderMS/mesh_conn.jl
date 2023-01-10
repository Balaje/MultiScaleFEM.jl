#################################################
# Function sto generate the connectivity matrices
#################################################
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
Function to obtain the elem arrays corresponding to the order of polynomial
"""
function new_connectivity_matrices(elem, (q,p))
    # Returns the H¹ (q-polynomial) and L² (p-polynomial)
    nel = size(elem,1)
    new_elem = Matrix{Int64}(undef, nel, q+1)
    new_elem_1 = Matrix{Int64}(undef, nel, p+1)
    for i=1:nel
        new_elem[i,:] = elem[i,1]+(i-1)*(q-1): elem[i,2]+i*(q-1)
        new_elem_1[i,:] = elem[i,1]+(i-1)*(p): elem[i,2]+i*(p)-1
    end
    new_elem, new_elem_1
end

"""
# Function to obtain the new multiscale method connectivity matrix. The connectivity depends on the support of the basis (parameter, l)  
"""
function new_connectivity_matrices(elems, p, l)
    #new_elem, _ = new_connectivity_matrices(elems, (p,p))
    _, new_elem = new_connectivity_matrices(elems, (q,p))
    new_elem_ms = Matrix{Int64}(undef,nel,(p+1)*(2l+1))
    fill!(new_elem_ms,0)
    for i=1:nel
        start = (new_elem[i,1]-(p+1)*l ≤ 0) ? 1 : new_elem[i,1]-(p+1)*l
        last = (new_elem[i,end]+(p+1)*l ≥ new_elem[nel,end]) ? new_elem[nel,end] : new_elem[i,end]+(p+1)*l
        new_elem_ms[i,1:length(start:last)] =  start:last
        # start = ((i-l) ≤ 0) ? 1 : (i-l)
        # last = ((i+l) ≥ nel) ? nel : (i+l)
        # elinds = vec(permutedims(new_elem[start:last,:],[2,1]))
        # new_elem_ms[i,1:length(elinds)] = elinds
    end
    new_elem_ms
end
  