import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from AdvectionDiffusion1D import AdvectionDiffusion1D
from expleja import expleja
from Integrators import expeuler

'''
Create Animation
'''

if __name__ == '__main__': 
    t_end = 1e-1
    Nx = 200
    Nt = 1000
    
    
    adv_coeff = 1e0 # Multiply advection matrix with adv_coeff
    dif_coeff = 1e1 # Multiply diffusion matrix with dif_coeff
    Pe = adv_coeff/dif_coeff
    A, u = AdvectionDiffusion1D(Nx, adv_coeff, dif_coeff, periodic = False, 
                                h = None, asLinearOp = False)
    u = u.flatten()
    uexakt = expleja(t_end,A,u.copy())
    
    x = np.linspace(0,1,len(u))
    
    t_list = np.linspace(0,0.1,Nt+1)
    y_list = np.zeros((Nt+1,len(u)))
    y_list[0] = u
    
    for k in range(Nt):
        y_list[k+1] += expeuler(A,t_list[k],y_list[k],t_list[k+1],1)[0]

    
    fig = plt.figure(1)
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    line, = ax.plot([], [], lw=2)
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,           
    
    # animation function.  This is called sequentially
    shift = 1# max(Nt//100,1)
    def animate(i):
        line.set_data(x, y_list[i*shift])
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   interval=100, repeat=True)

