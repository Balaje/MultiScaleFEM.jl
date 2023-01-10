using SparseArrays
using FastGaussQuadrature
using Plots
using ForwardDiff

include("1dFunctions.jl")
include("1dFunctionsMultiScale.jl")

# Initial Data
ε = 2^-8
u(x) = @. (x - x^2 + ε*(1/(4π)*sin(2π*x/ε) - 1/(2π)*x*sin(2π*x/ε) - ε/(4π^2)*cos(2π*x/ε) + ε/(4π^2)))
A(x) = @. (2 + cos(2π*x/ε))^(-1)
Dₓu(x) = ForwardDiff.derivative(u, x)
DₓADₓu(x) = @.  ForwardDiff.derivative(y -> A(y)*Dₓu(y), x)
f(x) = -DₓADₓu(x)
ug(x) = @. 0*x

order = Int(1/(ε)) # Number of points in local quadrature
Ω = (0,1)
gl = gausslegendre(order)
Ns = [2,4,8,16,32,64,128]
errs = zeros(Float64,length(Ns))
plts = Vector{Plots.Plot}(undef,length(Ns))

for (N,i) in zip(Ns,1:length(Ns))
    local nds, els = mesh(Ω, N)

    ## Visualize a sample local basis function and gradient [ ∈ (-1,1)]:
    # Λₘₛ¹(x) = φ̂ₘₛ(x, A, nds[els[1,:]]; order=order)[1];
    # Λₘₛ²(x) = φ̂ₘₛ(x, A, nds[els[1,:]]; order=order)[2];
    # ∇Λₘₛ¹(x) = ∇φ̂ₘₛ(x, A, nds[els[1,:]]; order=order)[1];
    # ∇Λₘₛ²(x) = ∇φ̂ₘₛ(x, A, nds[els[1,:]]; order=order)[2];
    # global plt = plot(LinRange(-1,1,600), Λₘₛ¹.(LinRange(-1,1,600)), color=:red, label="MS Basis 1")
    # plot!(plt, LinRange(-1,1,600), Λₘₛ².(LinRange(-1,1,600)), color=:blue, label="MS Basis 2")
    # plot!(plt, LinRange(-1,1,600), ∇Λₘₛ¹.(LinRange(-1,1,600)), color=:black, label="∇ (MS Basis 1)")
    # plot!(plt, LinRange(-1,1,600), ∇Λₘₛ².(LinRange(-1,1,600)), color=:magenta, label="∇ (MS Basis 2)")

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

    plts[i] = plot(nds, uu, color=:blue)
    plot!(plts[i], LinRange(0,1,300), u(LinRange(0,1,300)), color=:red)

    print("Done N=",N,".\n")
end

plt3 = plot(1 ./Ns, errs, scale=:log10, lw=3, ls=:dot, color=:black, label="\$ || u - u_h || \$")
plot!(plt3, 1 ./Ns, (1 ./Ns).^2, scale=:log10, lw=3, color=:blue, label="O(h²)")