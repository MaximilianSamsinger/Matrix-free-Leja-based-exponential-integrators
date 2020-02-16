from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy as sp
import scipy.sparse
from AdvectionDiffusion import AdvectionDiffusion2D
from expleja import expleja
from scipy.sparse import diags

s = 50
param = [0.0,1,0]
Nx = 20

t = 0 # Start time
t_end = 1e-1 # Final time

Adv, u = AdvectionDiffusion2D(Nx, 1, 0)
Dif, u = AdvectionDiffusion2D(Nx, 0, 1)


def F(u, param=param):
    f = 0.5*Dif@(u*(u+2))
    g = 2*u*(Adv@u)
    h = u*(u-0.5)
    return param[0]*f + param[1]*g + param[2]*h

def dF(u, param=param):
    u = u.flatten()
    df = Dif@diags((u+1))
    dg = 2*(diags(Adv@u) + diags(u)@Adv)
    dh = diags(2*u-0.5)
    return sp.sparse.csr_matrix(param[0]*df + param[1]*dg + param[2]*dh)

''' Compute reference solution '''
solver = sp.integrate.ode(lambda t,u: F(u))
solver.set_integrator('vode', method='bdf', order=5, atol=1e-16,
                      rtol=1e-16, nsteps=100000, with_jacobian=False)
solver.set_initial_value(u.copy(), 0)

while solver.successful() and solver.t<t_end:
    solver.integrate(t_end)
reference_solution = solver.y

x = np.linspace(0,1,Nx+2)[1:Nx+1]
X, Y = np.meshgrid(x,x)
w = u.copy()

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
dFuu = dF(u)@u
surf = ax.plot_surface(X,Y,r.reshape(Nx,Nx), cmap=cm.coolwarm)

fig = plt.figure()
ax = fig.gca(projection='3d')
dFuu = dF(u)@u
surf = ax.plot_surface(X,Y,dFuu.reshape(Nx,Nx), cmap=cm.coolwarm)

fig = plt.figure()
ax = fig.gca(projection='3d')
dFuu2 = (F(u+1e-6*u)-F(u))/1e-6
surf = ax.plot_surface(X,Y,dFuu2.reshape(Nx,Nx), cmap=cm.coolwarm)


print(np.linalg.norm(dFuu-dFuu2)/np.linalg.norm(dFuu))
'''
for k in range(s):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,w.reshape(Nx,Nx), cmap=cm.coolwarm)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  
    
    w += 0.1/s*F(w)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,reference_solution.reshape(Nx,Nx), cmap=cm.coolwarm)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')  
ax.set_title('Reference solution')