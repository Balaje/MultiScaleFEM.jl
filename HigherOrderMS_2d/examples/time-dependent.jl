include("./schur.jl")

"""
The Backward Difference Formula of order k for the linear heat equation
"""
function BDFk!(cache, tₙ::Float64, U::AbstractVecOrMat{T}, Δt::Float64, K::AbstractMatrix{T}, M::AbstractMatrix{T}, f!::Function, k::Int64) where T<:Real
  # U should be arranged in descending order (n+k), (n+k-1), ...
  @assert (size(U,2) == k) # Check if it is the right BDF-k
  dl_cache, fcache = cache
  coeffs = dl!(dl_cache, k)
  RHS = 1/coeffs[k+1]*(Δt)*(f!(fcache, tₙ+k*Δt))
  for i=0:k-1
    RHS += -(coeffs[k-i]/coeffs[k+1])*M*U[:,i+1]
  end
  LHS = (M + 1.0/(coeffs[k+1])*Δt*K)
  Uₙ₊ₖ = LHS\RHS
  # block_size = Int64(size(LHS,1)/ntimes)
  # Uₙ₊ₖ = BlockSchur(LHS, RHS, ntuple(i->block_size, ntimes))
  Uₙ₊ₖ
end
function get_dl_cache(k::Int64)
  0, 0, zeros(Float64,k+1)
end
function dl!(cache, k::Int64)
  sum, prod, res = cache
  fill!(res,0.0)
  xⱼ = 0.0:Float64(k)
  for i=1:k+1
    res[i] = dlₖ!((sum,prod), Float64(k), xⱼ, i)
  end
  res
end
function dlₖ!(cache, t::Float64, tⱼ::AbstractVector{Float64}, j::Int64)
  sum, prod = cache
  sum = 0.0
  prod = 1.0
  for l=1:lastindex(tⱼ)
    (l ≠ j) && begin
      prod = 1/(tⱼ[j]- tⱼ[l])
      for m=1:lastindex(tⱼ)
        (m ≠ j) && (m ≠ l) && begin
          prod = prod*(t - tⱼ[m])/(tⱼ[j]-tⱼ[m])
        end
      end
      sum += prod
    end
  end
  sum
end

"""
Function to setup the initial condition by evaluating the L² projection on the MS-space.
"""
function setup_initial_condition(u₀::Function, B::AbstractMatrix{T1}, fspace::FESpace; T=Float64) where T1<:Real
  massma = assemble_massma(fspace, x->1.0, 0; T=T)
  loadvec = assemble_loadvec(fspace, u₀, 4; T=T)
  Mₘₛ = B'*massma*B
  Lₘₛ = B'*loadvec
  collect(Mₘₛ)\Lₘₛ
end
