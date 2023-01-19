using SparseArrays
using FastGaussQuadrature
using Plots
using ForwardDiff
using LinearAlgebra

include("1dFunctions.jl")
include("1dFunctionsMultiScale.jl")

# Initial Data
ε = 2^-5
u(x) = @. (x - x^2 + ε*(1/(4π)*sin(2π*x/ε) - 1/(2π)*x*sin(2π*x/ε) - ε/(4π^2)*cos(2π*x/ε) + ε/(4π^2)))
A(x) = @. (2 + cos(2π*x/ε))^(-1)
Dₓu(x) = ForwardDiff.derivative(u, x)
DₓADₓu(x) = @.  ForwardDiff.derivative(y -> A(y)*Dₓu(y), x)
f(x) = -DₓADₓu(x)
ug(x) = @. 0*x

order = 120 # Number of points in local quadrature (Use higher order to capture the small oscillations)
Ω = (0,1)
gl = gausslegendre(order)
Ns = [2,4,8,16]
errs = zeros(Float64,length(Ns))
plts = Vector{Plots.Plot}(undef,length(Ns))

for (N,i) in zip(Ns,1:length(Ns))
    local nds, els = mesh(Ω, N)

    ## Visualize a sample local basis function and gradient [ ∈ (-1,1)] for 4 element case:
    if(N==4)
        global plt = plot()
        for i=1:N-1
            Λₘₛ¹(x) = φ̂ₘₛ(x, A, nds[els[i,:]]; order=order)[1];
            Λₘₛ²(x) = φ̂ₘₛ(x, A, nds[els[i+1,:]]; order=order)[2];
            plot!(plt, LinRange(nds[els[i,1]], nds[els[i,2]],600), Λₘₛ¹.(LinRange(nds[els[i,1]], nds[els[i,2]],600)), color=:red,label="")
            plot!(plt, LinRange(nds[els[i+1,1]], nds[els[i+1,2]],600), Λₘₛ².(LinRange(nds[els[i+1,1]], nds[els[i+1,2]],600)), color=:blue,label="")
        end
    end

    # Begin solution routine
    local tn = 1:size(nds,1)
    local bn = [1, size(nds,1)]
    local fn = setdiff(tn, bn)
    # Assemble the system of equations. 
    # (Optional parameter "local_matrix_vector_function" indicates whether 
    # the local system is a multiscale problem)
    local assembler = get_assembler(els)
    local MM, KK = assemble_matrix((assembler[1], assembler[2]), nds, A; 
                quad=gl, local_matrix_vector_function=local_matrix_vector_multiscale!)
    local FF = assemble_vector(assembler[3], nds, f; 
                quad=gl, local_matrix_vector_function=local_matrix_vector_multiscale!)
    # Apply the boundary conditions
    local bc = (KK[:, bn])*(ug.(nds[bn]))
    local uu = zeros(Float64, size(nds,1))
    # Solve the problem
    uu[fn] = (KK[fn,fn])\(FF[fn]-bc[fn])
    uu[bn] = ug.(nds[bn])
    errs[i] = l2err_multiscale(uu, u, nds, els, A; quad=gl)

    @show 

    plts[i] = plot(nds, uu, color=:blue, label="Approx. sol.")
    plot!(plts[i], LinRange(0,1,300), u(LinRange(0,1,300)), color=:red, label="Exact sol.")
    xlabel!(plts[i],"x")
    ylabel!(plts[i],"u(x) or uₕ(x)")

    print("Done N=",N,", Discrete l∞ error = "*string(norm(abs.(uu - u.(nds)),Inf))* "\n")
end

plt3 = plot(1 ./Ns, errs, scale=:log10, lw=3, ls=:dot, color=:black, label="\$ || u - u_h || \$")
plot!(plt3, 1 ./Ns, (1 ./Ns).^2, scale=:log10, lw=3, color=:blue, label="O(h²)")