# Abstract

In this master thesis we develop an algorithm to approximate the action of a matrix exponential function. In particular, we consider the case where the matrix is itself not available, but only its action as a linear operator. The construction is achieved by using a modified version of the real Leja method. We choose optimal interpolation parameters based on a spectral radius estimate computed by the power method. With this procedure we construct exponential Rosenbrock-type integrators to solve stiff advection-diffusion-reaction equations. We compare the performance of these integrators with other matrix-free differential equation solvers.

# Experiments
The main file for all experiments is Experiments.py. The results are saved as h5-files. The plots are computed using the corresponding ExperimentX_plots.py files.
We consider discretized advection-diffusion equations of the form

dt u = F(u) = Au + g(u), 
u(0) = u_0,

where A and g is a linear and nonlinear part of F respectively.

We solve these system with five different one-step methods. 
  1. rk2: The explicit midpoint method of order 2
  2. rk4: The classical Runge-Kutta method method of order 4
  3. cn2: The Crank-Nicolson method of order 2
  4. exprb2: The exponential Rosenbrock-Euler method of order 2.
  5. exprb4: An exponential integrator of order 4 
