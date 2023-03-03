##### ###### ###### ###### ###### ###### ###### ###### ####
# Module to implement solving multiple times efficiently  #
##### ###### ###### ###### ###### ###### ###### ###### ####

module SolveLinearSystem
  using LinearAlgebra
  using LoopVectorization

  function solution_cache(stima::AbstractMatrix{Float64}, fn::AbstractVector{Int64})
    K = lu(stima[fn,fn])
    solvec = zeros(Float64, length(fn))
    loadvec = zero(solvec)
    solvec, K, loadvec
  end

  function solve!(cache, f::AbstractVector{Float64}, bn::AbstractVector{Int64})
    solvec, luK, loadvec = cache
    fill!(loadvec,0.0)
    fill!(solvec,0.0)
    getindex!(loadvec, f, bn)
    ldiv!(solvec, luK, loadvec) # Allocates memory while solving
    solvec
  end

  function getindex!(cache, vec, inds)
    @turbo for i=1:lastindex(inds)
      cache[i] = vec[inds[i]]
    end
    cache
  end

  function get_solution(solcache, bvals)
    vcat(bvals[1],solcache[1],bvals[2])
  end

end