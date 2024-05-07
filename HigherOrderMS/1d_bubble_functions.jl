#### #### #### #### #### #### #### #### #### #### 
# Code to test the bubble function implementation 
#### #### #### #### #### #### #### #### #### #### 

domain = (0.0,1.0)
nc = 9
p = 3
nf = 2^15
nds_fine = LinRange(domain..., nf+1)
C = _c(domain, nc, p)
elem_coarse = [i+j for i=1:nc, j=0:1]
nds_coarse = LinRange(domain..., nc+1)

plt1 = Plots.plot();
plt2 = Plots.plot()
plt3 = Plots.plot();
plt4 = Plots.plot();

for t = [1,5]
  tri = Tuple(nds_coarse[elem_coarse[t,:]])
  start = max(1,t-1)
  last = min(nc,t+1)    
  if(t==1 || t==nc) 
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  else
    patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[t,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
  end  
  P = (patch[1][1], patch[end][2]); 

  Plots.plot!(plt1, nds_fine, Λₖ!.(nds_fine, Ref(tri), Ref(p), Ref(1)), label="\$ \\Lambda_{1,K} \$", title="Legendre Polynomials")
  Plots.plot!(plt2, nds_fine, bⱼ.(nds_fine, Ref(tri), Ref(C[end]), 1), label="\$ b_{1,K} \$", title="Bubble Functions")
  Plots.plot!(plt3, nds_fine, νⱼ.(nds_fine, t, Ref(C)), label="\$ \\nu_{K} \$", lw=1)
  Plots.plot!(plt3, nds_fine, ιⱼ.(nds_fine, Ref(tri), Ref(P)), label="\$ \\iota_{K} \$ ", lw=1, ls=:dash, title="Auxiliary functions")
  Plots.plot!(plt4, nds_fine, Pₕbⱼ.(nds_fine, t, Ref(C)), label="\$ \\iota_{K} + \\nu_{K} \$", title="Extended bubble functions")
end

using Test 
using LinearAlgebra

@testset "Check the L²-projection of the corrected bubble functions" begin
  for p=[1,2,3]
    C = _c(domain, nc, p)
    n = ceil(Int64, 0.5*(2*(2p+2)+1));
    x̂, w = gausslegendre(n);        
    Π = zeros(p+1, p+1);   
    for t=1:nc
      tri = Tuple(nds_coarse[elem_coarse[t,:]])
      xqs = (tri[2]+tri[1])/2 .+ (tri[2]-tri[1])/2*x̂  
      for i=1:p+1
        Π[i,i] = sum(w .* Λₖ!.(xqs, Ref(tri), p, i) .* Λₖ!.(xqs, Ref(tri), p, i))*(tri[2]-tri[1])*0.5        
      end

      # Test whether the Legendre polynomials are orthonormal      
      @test Π ≈ I(p+1) 

      # Get the patch
      start = max(1,t-1)
      last = min(nc,t+1)    
      if(t==1 || t==nc) 
        patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
      else
        patch = Tuple(nds_coarse[elem_coarse[start,:]]), Tuple(nds_coarse[elem_coarse[t,:]]), Tuple(nds_coarse[elem_coarse[last,:]])
      end  
      P = (patch[1][1], patch[end][2]); 

      # Compute the L² projection of the zero-th order bubble functions
      F1 = zeros(p+1);        
      for i=1:p+1
        F1[i] = sum(w .* bⱼ.(xqs, Ref(tri), Ref(C[end]), 1) .* Λₖ!.(xqs, Ref(tri), p, i))*(tri[2]-tri[1])*0.5                
      end

      # Compute the L² projection of the zero-th order extended bubble functions
      F2 = zeros(p+1);        
      for i=1:p+1
        F2[i] = sum(w .* Pₕbⱼ.(xqs, t, Ref(C)) .* Λₖ!.(xqs, Ref(tri), p, i))*(tri[2]-tri[1])*0.5                
      end

      function E1(p)
        res = zeros(p+1)
        res[1] = 1.0
        res
      end

      # Test if the L² projection of the extended bubble is equal to the Legendre polynomial      
      @test F1 ≈ E1(p)
      @test F2 ≈ E1(p)

    end
  end
end; # All tests should pass