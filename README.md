# MultiscaleFEM.jl

- [Introduction](#introduction)
- [Higher order multiscale method](#higher-order-multiscale-method)
  * [Poisson equation in 1D](#poisson-equation-in-1d)
  * [Heat equation in 1D](#heat-equation-in-1d)
  * [Wave equation in 1D](#wave-equation-in-1d)
  * [Rate of convergence of the multiscale method](#rate-of-convergence-of-the-multiscale-method)
    + [Poisson Equation](#poisson-equation)
      - [Smooth Diffusion Coefficients](#smooth-diffusion-coefficients)
      - [Oscillatory and Random Diffusion Coefficients](#oscillatory-and-random-diffusion-coefficients)
    + [Time dependent problems](#time-dependent-problems)
      - [Heat equation](#heat-equation)
        * [Smooth and Oscillatory Diffusion coefficient](#smooth-and-oscillatory-diffusion-coefficient)
        * [Random diffusion coefficient](#random-diffusion-coefficient)
      - [Wave Equation](#wave-equation)
        * [Constant wave speed](#constant-wave-speed)
        * [Smooth and varying wave speed](#smooth-and-varying-wave-speed)
        * [Oscillatory wave-speed without well-prepared data](#oscillatory-wave-speed-without-well-prepared-data)
        * [Oscillatory wave-speed with well-prepared data](#oscillatory-wave-speed-with-well-prepared-data)
        * [Highly-oscillatory wave-speed with well-prepared data](#highly-oscillatory-wave-speed-with-well-prepared-data)
        * [Highly-oscillatory wave-speed with well-prepared data solved for large final time](#highly-oscillatory-wave-speed-with-well-prepared-data-solved-for-large-final-time)
        * [Random oscillatory wave-speed](#random-wave-speed)
        * [Random oscillatory wave-speed solved for large final time](#random-wave-speed-solved-for-large-final-time)
- [Localized Orthogonal Decomposition Method](#localized-orthogonal-decomposition-method)
- [Implementation of the Higher Order Multiscale Method in two dimensions](#implementation-of-the-higher-order-multiscale-method-in-two-dimensions)
  * [The Coarse-To-Fine map](#the-coarse-to-fine-map)
  * [Patch information](#patch-information)
- [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

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

The implementation is based on the paper by [Maier, R.](https://epubs.siam.org/doi/abs/10.1137/20M1364321). This method is local upto the patch of size $l$ outside the element. I use the standard finite element method on the patch to compute the projection of the $L^2$ functions on the coarse space. This mesh on the fine scale needs to be sufficiently small to resolve the fine scale effects. Since the problem is solved locally, the fine-scale effects could be captured using reasonably large mesh on the patch. 

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

##### Heat equation

###### Smooth and Oscillatory Diffusion coefficient

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

###### Random diffusion coefficient

Finally, I test the problem for random-coefficients which are piecewise-constant on the scale $\epsilon = 2^{-12}$. 

`(p=1)` | `(p=2)` |
--- | --- |
![](./HigherOrderMS/Images/HeatEquation/ooc_p1_random_coeff_t1.0.png) | ![](./HigherOrderMS/Images/HeatEquation/ooc_p2_random_coeff_t1.0.png) | 

`(p=3)` | Random Diffusion Coefficient | 
--- | --- |
![](./HigherOrderMS/Images/HeatEquation/ooc_p3_random_coeff_t1.0.png) | ![](./HigherOrderMS/Images/HeatEquation/random_coeff.png) |


The method again shows optimal convergence for `p=1` but seem to slightly deteriorate for `p=2,3` as the mesh-size decreases. This can be seen in the case of the wave equation as well, which will be covered in the next section.

We perform a few more tests for the heat equation. Consider the following problem 

$$
\begin{align*}
  u_t - (A(x)u_x(x,t))_x = f(x,t) &\quad (x,t) \in (0,1) \times (0,T),\\
  u(x,0) = u_0(x) &\quad x \in (0,1),\\
  u(0,t) = u(1,t) =0 &\quad t \in (0,T),
\end{align*}
$$

I take the diffusion coefficient 

$$ 
A(x) = \text{rand}\left(0.5,1, \epsilon = 2^{-12}\right)
$$

and solve the problem using the multiscale method using `p=3`. I take the coarse mesh size $H=2^{-1}, 2^{-2}, 2^{-3}, \cdots 2^{-7}$ with the background fine scale discretization set at $h = 2^{-15}$. I use the 4th order Backward Difference Formula (BDF-4) to discretize the temporal direction. I take the time step size $\Delta t = 10^{-3}$ and solve till final time $T=0.5$ s. I consider the following cases:

1. **Case 1**
    $$
    f(x,t) = \sin (\pi x) \sin(\pi t), \quad u_0(x) = 0.
    $$

    We observe that the convergence rates are suboptimal for the higher order multiscale method `(p=3)` as $H \to 0$. However, the rate seems to be optimal for the first two mesh sizes. 

    `(p=3)` |
    --- |
    ![](./HigherOrderMS/Images/HeatEquation/ooc_p3_random_coeff_t0.5_u0_0_f.png) |

    ``` julia
    L²Error = 
    [6.904026225412479e-5, 
    9.994602736284981e-7, 
    3.897034713700309e-8, 
    3.802574439053447e-9, 
    9.080179587634707e-10, 
    1.3381403737114586e-10, 
    4.2318825691302157e-11, 
    3.8970415993978335e-11]

    log.(L²Error[2:end] ./ L²Error[1:end - 1]) ./ log(0.5) = 
    [6.110144910356789, 
    4.680700536317692, 
    3.357328387207551, 
    2.0661837538612366, 
    2.7624913657642014, 
    1.66085796597924, 
    0.1189202627442888]
    ```

2. **Case 2**
    $$
    f(x,t) = 0, \quad u_0(x) = \sin (\pi x).
    $$

    Again, we observe that the convergence rates are suboptimal for the higher order multiscale method `(p=3)` as $H \to 0$. However, the rate seems to be suboptimal earlier than observed in **Case 1**.

    `(p=3)` |
    --- |
    ![](./HigherOrderMS/Images/HeatEquation/ooc_p3_random_coeff_t0.5_f_0_u0_sin_pi_x.png) |

    The rate of convergence and the error magnitudes are as follows:   

    ```julia
    L²Error = 
    [1.7875148956715388e-5, 
    3.524163429947011e-7, 
    2.3963586522072257e-8, 
    7.319979871768478e-9, 
    1.219071871938366e-9, 
    1.7618198430902445e-10, 
    4.3432779965066374e-11, 
    2.448469254518588e-11]

    log.(L²Error[2:end] ./ L²Error[1:end - 1]) ./ log(0.5) = 
    [5.664530624433729, 
    3.8783650784416164, 
    1.7109322594020788, 
    2.586056497047787, 
    2.79064487191708, 
    2.0202102051473467, 
    0.8269042168959881]
    ```

3. **Case 3**

    However, the rate becomes optimal if I take a smooth diffusion coefficient or piecewise constant coefficients at a much coarser scale. Here I assume $f(x,t) = 0, \quad u_0(x) = \sin (\pi x)$.

    `(p=3)` Smooth Coefficient |
    --- |
    ![](./HigherOrderMS/Images/HeatEquation/ooc_p3_smooth_coeff_t0.5_f_0_u0_sin_pi_x.png) |

    ``` julia
    L²Error = 
    [1.0825524118533765e-8, 
    7.334129301856646e-10, 
    1.1287478539913153e-11, 
    2.1202768627351573e-13, 
    4.777345078223101e-15, 
    2.403254052690372e-14, 
    1.4162339474999647e-14, 
    2.3721930180271398e-14]

    log.(L²Error[2:end] ./ L²Error[1:end - 1]) ./ log(0.5) = 
    [3.8836673633874934, 
    6.021830551287917, 
    5.73432677243487, 
    5.471899762295737,
    -2.3307081719004294, 
    0.7629295629348244,
    -0.744161798592859]
    ```

##### Wave Equation

###### Constant wave speed

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

I observe that for constant wave speed case, the method converges with the optimal convergence rates. 

###### Smooth and varying wave speed

I now solve the problem with the following data

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

###### Oscillatory wave-speed without well-prepared data

Now I solve the same problem keeping the initial and boundary data same as Example 2, but with an oscillatory wave speed $c^2(x) = \left(0.25 + 0.125\cos\left(\frac{2\pi x}{2\times 10^{-2}}\right)\right)^{-1}$. Here I observe that the method does not show any convergence. This may be due to the initial data not being "well-prepared", which is an assumption to obtain optimal convergence rates. 

`(p=1)` | `(p=2)` |
--- | --- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p1_oscillatory_wave_speed.png)  | ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_oscillatory_wave_speed.png)  

`(p=3)` |
--- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p3_oscillatory_wave_speed.png)  | 

###### Oscillatory wave-speed with well-prepared data

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

This maybe due to the initial data being well-prepared (see [Abdulle, A. and Henning, P.](https://www.ams.org/journals/mcom/2017-86-304/S0025-5718-2016-03114-2/)).

###### Highly-oscillatory wave-speed with well-prepared data

To be extra sure, now I run the above problem with the same non-zero forcing and zero initial data, but with the wave speed

$$
c^2(x) = \left(0.25 + 0.125\cos\left(\frac{2\pi x}{2 \times 10^{-5}}\right)\right)^{-1}
$$

This gives a highly oscillatory wave-speed, which at a very fine scale looks like smooth function. I still seem to obtain optimal convergence, even for highly oscillatory wave speeds.

 Highly oscillating Wave Speed | `(p=1)` |
--- | --- | 
![](./HigherOrderMS/Images/WaveEquation/highly_osc_wave_speed.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p1_high_osc_forcing.png) |

| `(p=2)` | `(p=3)` |
--- | --- |
 ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_high_osc_forcing.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p3_high_osc_forcing.png) | 

###### Highly-oscillatory wave-speed with well-prepared data solved for large final time

Optimal convergence for the highly oscillatory case is observed even when we solve the problem for a long time. Here I show an example for the above problem with `p=3` at $T=7.5$ s. 

`(p=3)` | 
--- |
![](./HigherOrderMS/Images/WaveEquation/ooc_p3_osc_forcing_s7.5s.png) |

###### Random wave-speed

Next, I show the rate of convergence results for a random piecewise-constant in a on the scale $\epsilon = 2^{-12}$ with randomly chosen values in $[0.5,5]$. The choice of wave forcing and the initial conditions are the same as that of the well prepared case. I solve the problem till the final time reaches $T=1.5$ s. I generally observe optimal convergence in all the cases.

 Random Wave Speed | `(p=1)` |
--- | --- | 
![](./HigherOrderMS/Images/WaveEquation/random_wave_speed.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p1_random_forcing.png) |

| `(p=2)` | `(p=3)` |
--- | --- |
 ![](./HigherOrderMS/Images/WaveEquation/ooc_p2_random_forcing.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p3_random_forcing.png) |


###### Random wave-speed solved for large final time

Finally, I solve the problem with the random wave-speed till final time $T = 7.5$ s, and show the convergence rates for `p=1,3`. The rate of convergence seem optimal for `p=1`, but seems to slow down for the `p=3` case. The reference solution was obtained using the traditional finite element method on a very fine mesh $h=2^{-15}$ whereas the oscillations are on the scale $\epsilon = 2^{-12}$.

| `(p=1)` | `(p=3)` |
--- | --- |
 ![](./HigherOrderMS/Images/WaveEquation/ooc_p1_random_forcing_t7.5.png) | ![](./HigherOrderMS/Images/WaveEquation/ooc_p3_random_forcing_t7.5.png) | 

## Localized Orthogonal Decomposition Method
-------

The localized orthogonal decomposition method implementation can be found inside the `LOD/` folder. The program `LOD/main.jl` contains the code to check the rate of convergence of the LOD method. The file `LOD/1dFunctions.jl` contains the routines to compute the standard finite element basis along with the functions assemble the global matrices. The file `LOD/1dFunctionsMultiScale` contains the code to compute the multi-scale basis along with the function to compute the $L^2$ error of the multiscale-FEM solution. Running the code `LOD/main.jl`. The multiscale basis corresponding to $(H=0.25)$ and $\varepsilon=2^{-5}$ along with the finite element solution and the rate of convergence for various mesh-size is shown below:


![](./LOD/basis.png) | ![](./LOD/solutions.png) | 
--- | --- | 

![](./LOD/ooc_lod.png) | 
--- |

For more details on the method, refer to [Målqvist, A. et al](https://epubs.siam.org/doi/book/10.1137/1.9781611976458).

## Implementation of the Higher Order Multiscale Method in two dimensions

I began working on the 2D implementation of the higher-order multiscale method. The code is found in `2d_HigherOrderMS/` within the repository. 

**Please note that the current implementation can change a lot, since its mostly Work in Progress.**

### The Coarse-To-Fine map

Implementation in 2D involves a bit of geometrical pre-processing, which involves extracting the patch of all the elements in the coarse space along with the fine-scale elements present inside the patch. The central assumption in obtaining this information is that the fine-scale discretization is obtained from a uniform refinement of the coarse scale elements. 

At this stage, the current implementation has only the patch computations coded up. That too just for triangles! The complication here is the map between the coarse scale and fine scale map. For simple meshes involving simplices and hexahedrons, this map can be computed by exploiting an underlying pattern. Ideally, we need the underlying refinement strategy to obtain this map. I hard coded this inside the function 

```julia
get_coarse_to_fine_map(num_coarse_cells::Int64, num_fine_cells::Int64)
```

which relies on two other function to obtain this map. This interface is subject to change in the future.

### Patch information

Obtaining the patch on the coarse scale is relatively straightforward. We can define a metric that defines the "distance" between the elements and then search the `KDTree` built using the connectivity matrix of the fine scale mesh. The amazing `NearestNeighbors.jl` package contains the interface to implement `KDTree` searching. The definition of the metric is as follows

```julia
struct ElemDist <: NearestNeighbors.Distances.Metric end
function NearestNeighbors.Distances.evaluate(::ElemDist, x::AbstractVector, y::AbstractVector)
  dist = abs(x[1] - y[1])
  for i=1:lastindex(x), j=1:lastindex(y)
    dist = min(dist, abs(x[i]-y[j]))
  end
  dist+1
end
```

The method `evaluate` is extended to a user-defined DataType named `ElemDist` which accepts two vectors which are the local-to-global map of the elements in the fine-scale discretization, i.e., two rows of the finite element connectivity matrix. The method returns the distance between the elements which is defined as the minimum difference between all the elements of the input vectors. The result is an integer which is nothing but the patch size parameter $l$ in multiscale methods. We then search the KD-Tree until we find all elements in the coarse-scale such that `ElemDist` is less than or equal to $l$:

```julia
function get_patch_coarse_elem(ms_space::MultiScaleFESpace, l::Int64, el::Int64)
  U = ms_space.UH
  tree = ms_space.elemTree
  Ω = get_triangulation(U)
  σ = get_cell_node_ids(Ω)
  el_inds = inrange(tree, σ[el], 1) # Find patch of size 1
  for _=2:l # Recursively do this for 2:l and collect the unique indices. 
    X = [inrange(tree, i, 1) for i in σ[el_inds]]
    el_inds = unique(vcat(X...))
  end
  sort(el_inds)
  # There may be a better way to do this... Need to check.
end
```

This gives the indices of the coarse-scale elements present in the patch. The function 

```julia
function get_patch_triangulation(ms_spaces::MultiScaleFESpace, l::Int64, num_coarse_cells::Int64)
```

extracts the patch elements and then returns the element-patch wise discrete models. Then, the next step, which is the trickiest, is to obtain the fine scale discretization on the patch. A quick glance would just tell us that collecting the coarse scale elements and then applying the coarse-to-fine map will do it. However, to construct the discrete models in Gridap, we need the local ordering along with the nodal coordinates, which we do not have directly. I wrote this function 

```julia
function get_patch_triangulation(ms_space::MultiScaleFESpace, l::Int64, num_coarse_cells::Int64, coarse_to_fine_elems)
```

which extracts this information from the fine-scale discretization and returns the fine-scale version of the element-patch wise discrete models. I show the results below:

`l=1` patch of coarse scale elements 1, 10, 400, 405 |
--- |
![](./2d_HigherOrderMS/Images/2d_patch_trian.png) |
The visualization was done using Paraview. In the figure, you can see the coarse scale discrezation in the background along with the coarse scale patch, whose edges are highlighted in Red and Green. The background Blue coloured triangles denote the fine-scale discretization within the patch. |

`l=2` patch of coarse-scale elements 10,400 | 
--- | 
![](./2d_HigherOrderMS/Images/2d_patch_trian_l2.png) | 

`l=3` patch of coarse-scale elements 10,400 |
--- |
![](./2d_HigherOrderMS/Images/2d_patch_trian_l3.png) |

The next step is to solve the saddle point problems on the meshes and then obtain the multiscale bases! 

### The Multiscale bases

Multiscale Bases Function `(p=1)` with a random Diffusion Coefficient |
--- |
![](./2d_HigherOrderMS/Images/2d_ms_basis_1_patch_1_and_2.png) |

For the discontinuous $L^2$-functions `(p=1)`, I took the scaled monomials 

$$
\mathcal{M} = 1, \frac{x-x_D}{h_D}, \frac{y-y_D}{h_D}
$$

upto order $p$ on each triangles. Each triangle in the patch has 3 multiscale bases, obtained by projecting the function onto $H^1_0(N^l(K_i))$. The domain $N^l(K_i)$ denotes the $l$-sized patch of the element $K_i$ in the coarse mesh. The above picture shows the projection of the first scaled monomial $\mathcal{M}_i (x,y)= 1$ onto $H^1_0(N^1(K_1))$ (left) and $H^1_0(N^1(K_1))$ (right). Here $K_1$ and $K_2$ are the first two triangles in the coarse space. The diffusion coefficient in the problem is a random field constant on each element on the fine scale. The fine scale mesh is a triangulation with $32768$ elements. The coarse mesh is also a regular triangulation with much fewer, $32$ elements. The meshes are shown below. The oscillations seem to appear in the multiscale bases, which is the desired outcome of the projection. Please note, however, the code is still under developement and runs slow.

$l=1$ patch of the second element $K_2$ in the coarse scale (blue) along with the fine scale mesh (red)|
--- |
![](./2d_HigherOrderMS/Images/2d_patch_2_bg_fine.png) |

## References

- Målqvist, A. and Peterseim, D., 2020. Numerical homogenization by localized orthogonal decomposition. Society for Industrial and Applied Mathematics.
- Maier, R., 2021. A high-order approach to elliptic multiscale problems with general unstructured coefficients. SIAM Journal on Numerical Analysis, 59(2), pp.1067-1089.
- Abdulle, A. and Henning, P., 2017. Localized orthogonal decomposition method for the wave equation with a continuum of scales. Mathematics of Computation, 86(304), pp.549-587.