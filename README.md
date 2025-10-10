# MultiscaleFEM.jl

## Introduction

This repository contains the source code to implement the enhanced Higher Order Localized Orthogonal Decomposition (eho-LOD) method discussed in *Arxiv Link Here*. The code can be used to generate the results presented in the manuscript. In the manuscript, we consider solving the heat equation:

$$
\begin{align*}
  \partial_t u - \nabla\cdot\left(A_{\varepsilon}(x) \nabla u\right) = f(x,t), &\quad x\in\Omega, \quad t > 0,\\
  u(x,0) = u_0(x), &\quad x \in \Omega,\\
  u(x,t) = 0, &\quad x \in \partial\Omega, \quad t > 0,
\end{align*}
$$

where $A_{\varepsilon}(x)$ is an oscillatory diffusion coefficient at a scale $\varepsilon \ll 1$. It is well-known that the classical finite element method with mesh-size $h < \varepsilon$ is computationally expensive. Higher order methods based on the LOD framework [(Maier R., 2021)](https://epubs.siam.org/doi/10.1137/20M1364321) aim to construct basis functions on a coarse mesh $H \gg \varepsilon$ that contains the information about the fine-scale. This then leads to a computationally efficient method that depends on coarse-scale discretization $H$. However, for time-dependent problems, it was shown in [(Krumbeigel F. and Maier R., 2024)](https://academic.oup.com/imajna/article-abstract/45/4/2248/7759638?redirectedFrom=fulltext) that for oscillatory diffusion coefficients, the higher order LOD methods exhibit at most $O(H^2)$ convergence for any $p > 0$. In this work, we develop an **enhanced-higher-order** LOD method that, for oscillatory diffusion coefficients, shows the optimal rate $O(H^{p+2})$ in the energy norm.

The `README.md` file is divided into three sections, with the first and second sections discussing the minimum working examples in 1D and 2D, respectively. The third section will discuss scaling the 2D code to a HPC system and possible future work.

## 1D Example

We assume $\Omega = [0,1]$ and a randomly oscillating diffusion coefficient 

$$
A_{\varepsilon}(x) = \text{rand}\left(0.1, 1; \varepsilon=2^{-9}\right)
$$

and the right hand side

$$
f(x,t) = \begin{cases}
  0, & x < 0.5\\
  \sin(\pi x)\sin^5(t), & x \ge 0.5
\end{cases}.
$$

### How to run the code?

The code for the 1D example is located `HigherOrderMS_1d/examples/`. There are three main scripts: 
- `1d_Poisson.jl`: Julia code to solve the Poisson problem using the higher-order multiscale method.
- `1d_Heat.jl`: Code to solve the heat equation using higher-order multiscale method (HMM).
- `1d_Heat_Corrected.jl`: Code to solve the heat equation using the **enhanced-higher-order** multiscale method (eho-HMM).

To run the code, open terminal, change directory to `HigherOrderMS_1d/` and run

```zsh
>> cd /path/to/package/HigherOrderMS_1d/
>> julia --project=.
```

Once in the Julian prompt, type:

```julia
julia> include("examples/1d_Poisson.jl");

(1/h) 	 (1/H) 	 p 	 l 	 ||e||₀ 	 √(a(e,e))

2048 	 8 	 1 	 8 	 1.4678662504791758e-5 	 0.000754588781498337
```

Similarly for the heat equation:

```julia
julia> include("examples/1d_Heat.jl");

(1/h) 	 (1/H) 	 p 	 l 	 j 	 ||e||₀ 	 √(a(e,e))

2048 	 16 	 1 	 16 	 0 	 1.2198995337457969e-7 	 1.7556893929137127e-5

julia> include("examples/1d_Heat_Corrected.jl");

(1/h) 	 (1/H) 	 p 	 l 	 j 	 ||e||₀ 	 √(a(e,e))

2048 	 16 	 1 	 16 	 1 	 1.2708506354990224e-8 	 2.8928773641723775e-6
```

We observe that in case of the enhanced-higher-order multiscale method (`1d_Heat_Corrected.jl`), the error is almost an order of magnitude better! For larger values of $p$, we observed that the matrix becomes ill-conditioned. Since we are solving a 1D problem, to get around this problem, we use high-precision arithmetic. In `1d_Heat_Corrected.jl`, we can set the data type of the problem globally by assigning the variable `T₁` (line 17). We have an interface to export `\` for solving linear systems for two implementations of high-precision arithmetic: `Float128` from Quadmath.jl and `Double64` from DoubleFloats.jl.

## 2D Example

One of the most important features of the high-order multiscale methods is that the basis functions on the coarse scale can be precomupted and then saved to be used for testing multiple forcing functions. The main motivation for this is that in applications where the diffusion coefficient could be treated as a material property, the basis functions need not be re-computed everytime. For 2D especially, it is thus desirable to separate the computation, into different modules and use files to read/write the basis functions. Following are scripts in `HigherOrderMS_2d/examples` folder:

1. `2d_Heat_Params.jl`: Write all the problem parameters into a directory.
2. `2d_Heat_Ref_Sol.jl`: Compute the reference solution using the classical finite element method on the fine scale discretization $h$.
3. `2d_Heat_Basis.jl`: Compute the multiscale basis functions on the coarse scale and write the solution to the working directory.
4. `2d_Heat_MS_Solve.jl` and `2d_Heat_MS_Solve_No_Corr.jl`: Solve the heat equation using the multiscale basis generated in Step 3.

### How to run the code?

The code should be run in the same order as shown above. The workflow is also illustrated in `HigherOrderMS_2d/run_2d.sh`. Open terminal and run the script with the following arguments:

```zsh
>> cd /path/to/package/HigherOrderMS_2d/
>> PROJECT_NAME=TEST
>> ./run_2d.sh 256 2 1 2 1 $PROJECT_NAME
# :
# SOME PROGRESS BARS
# :
256      2       1       2       1       0.0012284297879164758   0.024945601525605033
```

## References

1. Felix Krumbiegel, Roland Maier, A higher order multiscale method for the wave equation, IMA Journal of Numerical Analysis, Volume 45, Issue 4, July 2025, Pages 2248–2273.

2. Maier R. A high-order approach to elliptic multiscale problems with general unstructured coefficients. SIAM Journal on Numerical Analysis. 2021;59(2):1067-89.
