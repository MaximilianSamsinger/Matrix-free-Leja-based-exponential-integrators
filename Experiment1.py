from expleja import expleja
from AdvectionDiffusion1D import AdvectionDiffusion1D
import numpy as np
import pandas as pd
from Integrators import Integrator, rk2, rk4, cn2, exprb2
from time import time, sleep
from itertools import product
from multiprocessing import Process, Lock
from solve_ODE import solve_ODE

'''
Linear Case: dt u = adv dxx u + dif dx u
On the interval [0,1] with homogeneous Dirichlet boundary conditions
for t in [0,0.1]

For different fixed time stepsizes and matrix dimension we calculate the error
and the cost for each integration method
'''

#!ptrepack -o Experiment1.h5 Experiment11.h5

'''
Nx... discretize x-axis into Nx-1 parts
s... number of substeps used when integrating
'''

'''
TODO:
    - Adapt the code to handle the nonlinear case as well
'''

def Linear_Advection_Diffusion_Equation(Nx, dif, asLinearOp, filename, lock):
    t = 0 # Start time
    t_end = 1e-1 # Final time

    A, u = AdvectionDiffusion1D(Nx, 1., dif, asLinearOp = False) # Dont change
    reference_solution, tmp = expleja(t_end, A, u)[:2] # Double precision

    target_errors = [2**-10,2**-24]

    if asLinearOp is True:
        exprb2.maxsubsteps = 1001
        substeps = [125, 250, 500, 1000]
        methods = [exprb2]
        powerits = [2,3,4,6,8,10,25,50]
    else:
        exprb2.maxsubsteps = len(tmp)*10
        substeps = (1.10**np.array(range(1,122))).astype('int')
        methods = [exprb2, cn2, rk2, rk4]
        powerits = [None]
    substeps = np.unique(substeps)
    A, u = AdvectionDiffusion1D(Nx, 1, dif, asLinearOp = asLinearOp)

    columns = ['substeps', 'Nx', 'dif', 'rel_error_2norm', 'mv', 'misc']
    integrators = [Integrator(method, reference_solution, target_errors,
                              powerits, columns) for method in methods]

    integrator_inputs = [A, u, t, t_end]
    add_to_row = [Nx, dif]

    solve_ODE(integrators, integrator_inputs, substeps, add_to_row,
              filename, lock=lock)


if __name__ == '__main__':
    Nxlist = list(range(50,401,50))

    difs = [1e0, 1e-1, 1e-2] # Coefficient of diffusion matrix. Should be <= 1
    asLinearOp = True
    multiprocessing = True

    problem_to_solve = Linear_Advection_Diffusion_Equation
    filename = 'Experiment1LinOp.h5' if asLinearOp else 'Experiment1.h5'

    begin = time()
    if multiprocessing:
        lock = Lock()
        all_processes = [Process(target = problem_to_solve,
                                 args=(Nx, dif, asLinearOp, filename, lock)
                                 ) for Nx in Nxlist for dif in difs]

        for p in all_processes: p.start()
        for p in all_processes: p.join()

    else:
        for Nx, dif in product(Nxlist, difs):
            problem_to_solve(Nx, dif, asLinearOp, filename, None)

    pd.set_option('io.hdf.default_format','table')
    with pd.HDFStore(filename) as hdf:
        for key in hdf.keys():
            hdf[key] = hdf[key].drop_duplicates()
            print('Key',key)
            print(hdf[key])
    print('Total time:', time()-begin)
    sleep(120)