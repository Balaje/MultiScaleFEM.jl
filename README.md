# MultiscaleFEM.jl

## Introduction

This repository contains the source code to implement the Localized Orthogonal Decomposition method and the Higher order Multiscale Method to solve the Poisson problem

$$
\begin{align*}
    -(D_{\varepsilon}(x)u'(x))' = f(x) &\quad x \in \Omega = (0,1),\\
    u(0) = u(1) = 0,
\end{align*}
$$

the heat equation supplemented with initial and boundary conditions 

$$
\begin{align*}
  u_t(x,t) - (D_{\varepsilon}(x)u_x(x,t))_x = f(x,t) &\quad (x,t) \in (0,1) \times (0,\infty),\\
  u(x,0) = u_0(x) &\quad x \in (0,1),\\
  u(0,t) = u(1,t) =0 &\quad t \in (0,\infty),
\end{align*}
$$

and the wave equation

$$
\begin{align*}
  u_{tt}(x,t) - (D_{\varepsilon}(x)u_x(x,t))_x = f(x,t) &\quad (x,t) \in (0,1) \times (0,\infty),\\
  u(x,0) = u_0(x) &\quad x \in (0,1),\\
  u_t(x,0) = u_1(x) &\quad x \in (0,1),\\
  u(0,t) = u(1,t) =0 &\quad t \in (0,\infty),
\end{align*}
$$

Here $D_{\varepsilon}$ is a highly oscillatory coefficient. Traditional methods often need very small meshes to resolve the oscillations on the fine scale and thus we turn to multiscale methods.

## Higher order multiscale method
-------

The implementation is based on the paper by [Maier, R.](https://epubs.siam.org/doi/abs/10.1137/20M1364321). This method is local upto the patch of size $l$ ouside the element. I use the standard finite element method on the patch to compute the projection of the $L^2$ functions on the coarse space. This mesh on the fine scale needs to be sufficiently small to resolve the fine scale effects. Since the problem is solved locally, the fine-scale effects could be captured using reasonably large mesh on the patch. 

The new basis function then contains the fine scale information and can be used to find the numerical solution that contains the information on the fine scale. This needs to be computed once and can be used repeatedly, for example, to solve time dependent problems. For example, the following figure shows the multiscale basis function containing the fine scale information. 

| Smooth Diffusion Coefficient | Oscillatory Coefficient | 
| --- | --- |
| ![](./HigherOrderMS/Images/basis_el_1.png) | ![](./HigherOrderMS/Images/basis_el_1_osc.png)

The smooth diffusion coefficient does not contain any oscillations and hence the multiscale bases are smooth. If the diffusion coefficient is oscillatory, then the information is captured by the multiscale bases function. The diffusion coefficient for the oscillatory case here is assumed to be

$$
D(x) = \left(1 + 0.8\cos\left(\frac{2\pi x}{2^{-5}}\right)\right)^{-1}
$$

whereas the constant diffusion is 

$$
D(x) = 1
$$

### Poisson equation in 1D

The script `HigherOrderMS/1d_Poisson_eq.jl` contains the code to solve the one-dimensional Poisson problem using the Higher Order Multiscale method. I show the results three different diffusion coefficients:

$$
D_{\varepsilon}(x) = 0.5, \quad D_{\varepsilon}(x) = \left(2 + \cos{\frac{2\pi x}{2^{-6}}}\right)^{-1}, \quad D_{\varepsilon}(x) = \text{rand}\left(0.5, 5.0;\, \epsilon = 2^{-12} \right).
$$

where $\epsilon = 2^{-12}$ denotes the scale of the randomness, i.e, the diffusion coefficient is constant at an interval of size $\epsilon$. We can observe that the multiscale method captures the exact solution accurately at small scales using relatively small elements `(N=8)` in the coarse space. The fine-scale mesh size was taken to be equal to $h = 2^{-15}$.

| Smooth Diffusion Term | Oscillatory Diffusion Term | 
| --- | --- |
| ![Smooth Diffusion Coefficient](./HigherOrderMS/Images/PoissonEquation/sol_5_smooth.png) | ![Oscillatory Diffusion Coefficient](./HigherOrderMS/Images/PoissonEquation/sol_6_oscillatory.png) |

Random Diffusion Term |
--- |
![Random Diffusion Coefficient](./HigherOrderMS/Images/PoissonEquation/sol_7_random.png) | 


### Heat equation in 1D

The script `HigherOrderMS/1d_heat_equation.jl` contains the code to solve the transient heat equation in 1D. The spatial part is handled using the finite element method (both traditional and multiscale) and the temporal part is discretized using the fourth order backward difference formula (BDF4). I use $h = 2^{-11}$ on the fine scale and $H=2^{-1}$ on the coarse scale, the (oscillatory) diffusion coefficient $D_{\varepsilon}(x)$ equal to

$$
D_{\varepsilon}(x) = \left(2 + \cos{\frac{2\pi x}{2^{-2}}}\right)^{-1}
$$

and the (smooth) coefficient 

$$
D_{\varepsilon}(x) = D_0 = 0.5
$$

In both cases, the right hand side $f(x,t) = 0$ and the initial condition $u_0(x) = \sin{\pi x}$. In the constant diffusion case, the exact solution can be obtained analytically and is equal to $u(x,t) = \exp{\left(-D_0 \pi^2 t\right)}u_0(x)$. Following figure shows the solution obtained using the multiscale method.  

| Smooth Diffusion Term | Oscillatory Diffusion Term |
| --- | --- |
| ![Smooth Diffusion Coefficient](./HigherOrderMS/Images/HeatEquation/heat_eq_all.png) | ![Smooth Diffusion Coefficient](./HigherOrderMS/Images/HeatEquation/heat_eq_all_osc.png) |


### Wave equation in 1D

The script `HigherOrderMS/1d_wave_equation.jl` contains the code to solve the acoustic wave equation in 1D. The spatial part is handled using the multiscale finite element method and the temporal part is discretized using Crank Nicolson scheme. I check two different wave speeds

$$
D_{\varepsilon}(x) = 4.0, \quad D_{\varepsilon}(x) = \left(0.25 + 0.125\cos\left(\frac{2\pi x}{2\times 10^{-2}}\right)\right)^{-1}
$$

In both cases, I set the right hand side $f(x,t) = 0$, the initial conditions $u(x,0) = 0$, $u_t(x,0) = 4\pi \sin\left(2\pi x\right)$. For the smooth wave speed case, the exact solution is given by $u(x,t) = \sin\left(2\pi x\right) \sin\left(4\pi t\right)$. We observe that the multiscale method gives a good approximation to the exact solution (smooth wave speed).

| Smooth wave speed | Oscillatory wave speed |
| --- | --- |
| ![](./HigherOrderMS/Images/WaveEquation/wave_eq_smooth.png) | ![](./HigherOrderMS/Images/WaveEquation/wave_eq_osc.png) |


### Rate of convergence of the multiscale method
-------

All the rate of convergence examples can be found inside the folder `HigherOrderMS/`. 

#### Poisson Equation

The script `HigherOrderMS/1d_rate_of_convergence_Poisson.jl` contains the code to study the convergence rates for the Poisson equation. To obtain the rate of convergence plots, we always assume that the exact solution is obtained by solving the problem using the traditional finite element method. This is because, in majority of the examples considered here, the exact solution is not known.

##### Smooth Diffusion Coefficients

The following figure shows the rate of convergence of the multiscale method for the lowest order case (`p=1` in the discontinuous space) and varying patch size, $l$. The example was run for a smooth diffusion coefficient. Following is the test example:

$$
 -(A(x)u'(x))' = f(x) \quad in \quad x \in \Omega = (0,1),
$$

with 

$$
  A(x) = 1, \quad f(x) = \pi^2\sin(\pi x)
$$

The corresponding exact solution is $u(x) = \sin(\pi x)$. 

![](./HigherOrderMS/Images/PoissonEquation/ooc_1.png) | 
--- |

We observe optimal convergence rates discussed in Maier, R., 2021 until the mesh size becomes too small. In that case a larger patch size (indicated by the parameter $l$) is required to obtain similar convergence rates for finer mesh. The growing part in the error is controlled by an $exp(-C_{dec} l)$ term and vanishes for higher value of $l$. 

![](./HigherOrderMS/Images/PoissonEquation/ooc_3.png) | 
--- |

This is in line with the observation made in Maier, R., 2021. Similar observations can be made for the higher-order cases as well, `(p=2)` and `(p=3)`. 

`(p=2)` | `(p=3)` |
--- | --- |
![](./HigherOrderMS/Images/PoissonEquation/ooc_2.png) | ![](./HigherOrderMS/Images/PoissonEquation/ooc_10_p3.png) |

We can solve the problem upto the coarse-mesh size $H = 2^0, 2^{-1}, \cdots, 2^{-12}$ with the fine scale at $h=2^{-16}$. However, the method does not show convergence for very fine coarse-meshes unless the localization parameter is chosen high enough.

![](./HigherOrderMS/Images/PoissonEquation/ooc_5.png) |
--- |

##### Oscillatory and Random Diffusion Coefficients

Finally we can observe the same behaviour for the other choices of diffusion coefficients. The diffusion coefficients were chose identical to the ones discussed in the previous section. The right hand side data $f(x) = \frac{\pi^2}{2}\sin{\pi x}$ for the oscillatory case and $f(x) = \sin{5\pi x}$ for the random diffusion case.

Oscillatory coefficient | Random coefficients |
--- | --- |
![](./HigherOrderMS/Images/PoissonEquation/ooc_6_oscillatory.png) | ![](./HigherOrderMS/Images/PoissonEquation/ooc_7_random_coeff.png) | 

#### Time dependent problems
-------

##### **Heat equation**

I solve the following parabolic initial boundary value problem using the multiscale method `(HigherOrderMS/rate_of_convergence_Heat_Equation.jl)`.

$$
\begin{align*}
  u_t - (A(x)u_x(x,t))_x = 0 &\quad (x,t) \in (0,1) \times (0,T),\\
  u(x,0) = \sin(\pi x) &\quad x \in (0,1),\\
  u(0,t) = u(1,t) =0 &\quad t \in (0,T),
\end{align*}
$$

I take $h = 2^{-16}$ and $H = 2^0, 2^{-1}, \cdots, 2^{-7}$. In the temporal direction, I set $\Delta t = 10^{-3}$ and solve till final time $T = 1.0$. I use the fourth order backward difference formula for discretizing the temporal part. The exact solution was taken to be the standard finite element solution on a mesh whose size is $h$. I compute the rate of convergence for the smooth diffusion coefficient 

$$
A(x) = 1.0,
$$

and the oscillatory coefficient

$$
A(x) = \left(2 + \cos{\frac{2\pi x}{2^{-6}}}\right)^{-1} .
$$

Constant coefficient | Oscillatory coefficient |
--- | --- |
![](./HigherOrderMS/Images/HeatEquation/ooc_8_heat_eq.png) | ![](./HigherOrderMS/Images/HeatEquation/ooc_9_heat_eq_osc.png)

Similar behavior can be seen for higher order methods also.

`(p=2)` (Oscillatory coefficient) | `(p=3)` (Oscillatory coefficient) |
--- | --- |
![](./HigherOrderMS/Images/HeatEquation/ooc_11_heat_eq_osc_p2.png) | ![](./HigherOrderMS/Images/HeatEquation/ooc_12_heat_eq_osc_p3.png) |


##### **Wave Equation**

I solve the following wave equation along with the prescribed initial and boundary conditions

$$
\begin{align*}
  u_{tt} - \left(c^2(x)u'(x,t)\right)' = 0 &\quad (x,t) \in (0,1) \times (0,T),\\
  u(x,0) = 0 &\quad x \in (0,1),\\
  u_t(x,0) = \pi \sin(\pi x) &\quad x \in (0,1),\\
  u(0,t) = u(1,t) =0 &\quad t \in (0,T),
\end{align*}
$$

using the multiscale method in space and Crank-Nicolson method in time. For the temporal discretization, I assume $\Delta t = 10^{-3}$ and solve till final times $T = 1.5$ s. Here I consider three wave speeds

Constant Wave Speed | Smooth Wave Speed |
--- | --- |
![](./HigherOrderMS/Images/WaveEquation/constant_wave_speed.png) | ![](./HigherOrderMS/Images/WaveEquation/smooth_wave_speed.png) | 

Oscillatory Wave Speed | 
--- |
![](./HigherOrderMS/Images/WaveEquation/oscillatory_wave_speed.png) |

First, I assume that the wave speed $c(x) = 1.0$. The exact solution is assumed to be the numerical solution obtained using the standard finite element method on a fine mesh of size $h=2^{-15}$. I take the coarse mesh size $H = 2^0, 2^{-1}, \cdots, 2^{-6}$ to study the convergence rates. Following plots show the rate of convergence of the multiscale method in space for `(p=1,2,3)`:

`(p=1)` | `(p=2)` | 
--- | --- | 
![](./HigherOrderMS/Images/WaveEquation/ooc_p1_constant_wave_speed.png)  | ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_constant_wave_speed.png)  | 

`(p=3)` |
--- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p3_constant_wave_speed.png)  | 

I observe that for constant wave speed case, the method converges with the optimal convergence rates. I now solve the problem with the following data

$$
\begin{align*}
  u_{tt} - \left(c^2(x)u'(x,t)\right)' = 0 &\quad (x,t) \in (0,1) \times (0,T),\\
  u(x,0) = 0 &\quad x \in (0,1),\\
  u_t(x,0) = \pi \sin(\pi x) &\quad x \in (0,1),\\
  u(0,t) = u(1,t) =0 &\quad t \in (0,T),
\end{align*}
$$

with a smooth, but non-constant wave speed $c^2(x) = \left(0.25 + 0.125\cos\left(\pi x\right)\right)^{-1}$ (shown above). Again, I observe optimal convergence rates when I solve till $T=1.5$ s.

`(p=1)` | `(p=2)` |
--- | --- | 
![](./HigherOrderMS/Images/WaveEquation/ooc_p1_smooth_wave_speed.png)  | ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_smooth_wave_speed.png)  | 

`(p=3)` |
--- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p3_smooth_wave_speed.png)  | 

Now I solve the same problem keeping the initial and boundary data same, but with an oscillatory wave speed $c^2(x) = \left(0.25 + 0.125\cos\left(\frac{2\pi x}{2\times 10^{-2}}\right)\right)^{-1}$. Here I observe that the method does not show any convergence. This may be due to the initial data not being "well-prepared", which is an assumption to obtain optimal convergence rates. 

`(p=1)` | `(p=2)` |
--- | --- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p1_oscillatory_wave_speed.png)  | ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_oscillatory_wave_speed.png)  

`(p=3)` |
--- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p3_oscillatory_wave_speed.png)  | 

However, if I consider this problem

$$
\begin{align*}
  u_{tt} - \left(c^2(x)u'(x,t)\right)' = f(x,t) &\quad (x,t) \in (0,1) \times (0,T),\\
  u(x,0) = 0 &\quad x \in (0,1),\\
  u_t(x,0) = 0 &\quad x \in (0,1),\\
  u(0,t) = u(1,t) =0 &\quad t \in (0,T),
\end{align*}
$$

with 

$$
c^2(x) = \left(0.25 + 0.125\cos\left(\frac{2\pi x}{2^{-5}}\right)\right)^{-1}, \quad f(x,t) = \sin(\pi x)\sin(t),
$$

and solve the problem till $T=1.5$ s, we observe the following convergence rates.

`(p=1)` | `(p=2)` 
--- | --- 
![](./HigherOrderMS/Images/WaveEquation/ooc_p1_osc_forcing.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_osc_forcing.png) 

| `(p=3)` |
| --- |
| ![](./HigherOrderMS/Images/WaveEquation/ooc_p3_osc_forcing.png) | 

To be extra sure, now I run the above problem with the same non-zero forcing and zero initial data, but with the wave speed

$$
c^2(x) = \left(0.25 + 0.125\cos\left(\frac{2\pi x}{2 \times 10^{-5}}\right)\right)^{-1}
$$

This gives a highly oscillatory wave-speed, which at a very fine scale looks like smooth function. I still seem to obtain optimal convergence, even for highly oscillatory wave speeds.

 Wave Speed | `(p=1)` |
--- | --- | 
![](./HigherOrderMS/Images/WaveEquation/highly_osc_wave_speed.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p1_high_osc_forcing.png) |

| `(p=2)` | `(p=3)` |
--- | --- |
 ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_high_osc_forcing.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p3_high_osc_forcing.png) | 

 Optimal convergence for the highly oscillatory case is observed even when we solve the problem for a long time. Here I show an example for the above problem with `p=3` at $T=7.5$ s. 

`(p=3)` | 
--- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p3_osc_forcing_s7.5s.png) |

## Localized Orthogonal Decomposition Method
-------

The localized orthogonal decomposition method implementation can be found inside the `LOD/` folder. The program `LOD/main.jl` contains the code to check the rate of convergence of the LOD method. The file `LOD/1dFunctions.jl` contains the routines to compute the standard finite element basis along with the functions assemble the global matrices. The file `LOD/1dFunctionsMultiScale` contains the code to compute the multi-scale basis along with the function to compute the $L^2$ error of the multiscale-FEM solution. Running the code `LOD/main.jl`. The multiscale basis corresponding to $(H=0.25)$ and $\varepsilon=2^{-5}$ along with the finite element solution and the rate of convergence for various mesh-size is shown below:


![](./LOD/basis.png) | ![](./LOD/solutions.png) | 
--- | --- | 

![](./LOD/ooc_lod.png) | 
--- |

For more details on the method, refer to [Målqvist, A. et al](https://epubs.siam.org/doi/book/10.1137/1.9781611976458).


## References

- Målqvist, A. and Peterseim, D., 2020. Numerical homogenization by localized orthogonal decomposition. Society for Industrial and Applied Mathematics.
- Maier, R., 2021. A high-order approach to elliptic multiscale problems with general unstructured coefficients. SIAM Journal on Numerical Analysis, 59(2), pp.1067-1089.
