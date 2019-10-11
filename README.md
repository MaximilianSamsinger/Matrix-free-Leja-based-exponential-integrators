Consider the one-dimensional linear advection-diffusion equation

dt u = dx u + 1/Pe dxx u, 
u(0) = u_0

on the interval [0,1] with homogenous Dirichlet boundary conditions. With Pe > 0 we denote the grid Peclet number. 
After a finite differences space discretization we get 

dt u = Au, 

where A is a linear Operator.

We solve this system with four different one-step methods. 
  1. rk2: The explicit midpoint method of order 2
  2. rk4: The classical Runge-Kutta method method of order 4
  3. cn2: The Crank-Nicolson method of order 2
  4. exprb2: The exponential Rosenbrock-Euler method of order 2.

Note that the exponential Euler method yields the exact solution for this problem. 
