from expleja import expleja
from AdvectionDiffusion1D import AdvectionDiffusion1D
import numpy as np
import pandas as pd
from Integrators import MatrixIntegrator, rk2_matrixinput,\
                        rk4_matrixinput, crankn, expeuler

from time import time
'''
Linear Case: dt u = dx u + dxx u
On the interval [0,1] with dirichlet homogeneous Dirichlet boundary conditions,
for t in [0,0.1]

For different fixed time stepsizes and matrix dimension we calculate the error
and the cost for each integration method
'''

'''
Todos
Change dataframe index?
Animate integration
'''


if __name__ == '__main__':
    t = 0 # Start time
    t_end = 1e-1 # Final time

    Nxlist = list(range(50,51,50)) # x-axis gets divided in Nx subintervals
    Ntlist = np.arange(5,100) # time-axis gets divided in Nt subintervals
    
    names = ['rk2', 'rk4', 'CrankNicolson', 'ExponentialEuler']
    methods = [rk2_matrixinput, rk4_matrixinput, crankn, expeuler] 
    
    columns = ['Integrator', 'Nt', 'Nx', 'Pe', 'error', 'mv_prod', 
               'other_costs', 'adv_coeff', 'dif_coeff']
    df = pd.DataFrame(columns=columns)
    
    for Nx in Nxlist:
        print('Nx:', Nx)
        adv_coeff = 1e0 # Multiply advection matrix with adv_coeff
        dif_coeff = 1e0 # Multiply diffusion matrix with dif_coeff
        Pe = adv_coeff/dif_coeff
        A, u = AdvectionDiffusion1D(Nx, adv_coeff, dif_coeff, periodic = False, 
                                    h = None, asLinearOp = False)
        
        exact_solution = expleja(t_end, A, u)[0]

        
        integrators = [MatrixIntegrator(name, method, Ntlist, exact_solution
                                 ) for name, method in zip(names, methods)]
        
        lst = []
        begin = time()
        for integrator in integrators:
            print(integrator.name)
            for Nt in Ntlist:
                _, error, functionEvaluations, otherCosts = \
                                        integrator.solve(A, t, u, t_end, Nt)
                
                lst.append([integrator.name, Nt, Nx, Pe, error, 
                        functionEvaluations, otherCosts, adv_coeff, dif_coeff])
                if error < 1e-9:
                    break
        df = df.append(pd.DataFrame(lst, columns=columns), ignore_index=True)
        print(time()-begin)
        
    print(df[df.columns.difference(['adv_coeff', 'dif_coeff','other_costs'])])
    
    exp = df[df['Integrator'] == 'ExponentialEuler']
    exp.logyplot(x = 'Nt', y = 'error', logy = True)
    '''
    tollist = [10**-k for k in range(4,9)]
    '''