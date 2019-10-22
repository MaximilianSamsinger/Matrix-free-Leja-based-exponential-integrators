from AdvectionDiffusion1D import AdvectionDiffusion1D
import numpy as np
import os
#For plotting
import pylab
from matplotlib import pyplot as plt

'''
Plot eigenvalues of Advection-Diffusion Matrix for different Peclet numbers
'''

diffusionlist = [1e0, 1e-1, 1e-2]
Nxlist = [50,400]

t_end = 1e-1


fig, axes = plt.subplots(1,len(Nxlist), figsize=(10, 4.8))
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()

periodic = False
save = True

for j, ax in enumerate(axes.flatten()):
    Nx = Nxlist[j]
    for k, dif in enumerate(diffusionlist):
        Pe = 1./dif
        print("Pe = ",Pe)

        A, u = AdvectionDiffusion1D(Nx, 1, dif, periodic=periodic)
        A = A.toarray().astype(np.complex_)
        w, _ = np.linalg.eig(A)

        ax.plot(np.real(w),np.imag(w),'.', label=f"Pe = {Pe}")

        ax.set_title(f'N = {Nx}')
        x1,x2,y1,y2 = ax.axis()
        y1,y2 = (y1, y2) if periodic else (-1,1)
        ax.axis((x1,x2,y1,y2))
        ax.set_xlabel('real')
        ax.set_ylabel('imaginary')
        ax.locator_params(axis='x', nbins=4)

pylab.legend(loc='lower left')
fig.suptitle('Spectrum of $A$')


name = 'Figures' + os.sep + 'Spectrum' + os.sep
name += 'periodic' if periodic else 'dirichlet'
name += '.pdf'
if save:
    plt.savefig(name, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()

print('*_______________*')