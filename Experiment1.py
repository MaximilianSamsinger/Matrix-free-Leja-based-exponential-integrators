from expleja import expleja
from AdvectionDiffusion1D import AdvectionDiffusion1D
import numpy as np
import pandas as pd
from Integrators import MatrixIntegrator, rk2_matrixinput,\
                        rk4_matrixinput, crankn, expeuler

from time import time
'''
Linear Case: dt u = adv dxx u + dif dx u 
On the interval [0,1] with dirichlet homogeneous Dirichlet boundary conditions,
for t in [0,0.1]

For different fixed time stepsizes and matrix dimension we calculate the error
and the cost for each integration method
'''

'''




'''

t = 0 # Start time
t_end = 1e-1 # Final time


#!ptrepack -o Experiment1.h5 Experiment11.h5

if __name__ == '__main__':    
    ''' Runtime warnings kill performance and will not lead to results '''
    np.seterr(over='raise', invalid='raise') # Raise error instead of warning
    
    adv = 1e0 # Multiply advection matrix with adv. Should be <= 1
    dif = 1e0 # Multiply diffusion matrix with dif. Should be <= 1
    
    '''
    Nx... discretize x-axis into Nx parts 
    Nt... Integrate with Nt substeps
    '''
    Nxlist = list(range(25,401,25))
    Ntlist = np.floor(1.12**np.array(range(1,90)))
    Ntlist = np.unique(Ntlist.astype('int'))
    
    names = ['crankn']
    methods = [crankn]
    columns = ['Nt', 'Nx', 'adv', 'dif', 
          'abs_error_2', 'rel_error_2', 'abs_error_inf', 'rel_error_inf',
          'mv', 'other_costs', 'gmres_tol']
    target_errors = [10**-k for k in range(2,9)]
        
    for Nx in Nxlist:
        print('Nx:', Nx)
        A, u = AdvectionDiffusion1D(Nx, adv, dif, periodic = False, 
                                    h = None, asLinearOp = False)
        
        exact_solution = expleja(t_end, A, u)[0] #Note: double precision

        
        integrators = [MatrixIntegrator(name, method, exact_solution
                                 ) for name, method in zip(names, methods)]
        
        integrator = integrators[0]
        begin = time()
        for target_error in target_errors:
            crankn.parameter = target_error/500
            ''' 
            Create and fill pandas dataframe 
            '''
            df = []
            print(integrator.name, target_error)
            ERROR_SMALL_ENOUGH = False #Skip when results get sufficiently bad
            for Nt in Ntlist:
                ''' 
                Calculate error and costs
                '''
                begin2 = time()
                try:
                    ''' Calculate error and costs'''
                    _, errors, costs = integrator.solve(A, t, u, t_end, Nt)
                    df.append([Nt, Nx, adv, dif] + list(errors) + list(costs)
                    + [crankn.parameter])
                    '''
                    If the error is small enough, decrease gmres tolerance
                    until the error is no longer small enough. 
                    (Find optimal gmres tolerance for a given target_error)
                    '''
                    if errors[3] < target_error:
                        crankn.parameter /= 1.1
                        _, errors, costs = integrator.solve(
                                A, t, u, t_end, Nt)
                        df.append([Nt, Nx, adv, dif] 
                            + list(errors) + list(costs)
                            + [crankn.parameter])
                        if errors[3] > target_error or (
                                crankn.parameter == target_error):
                            break
                        break

                except (FloatingPointError):
                    pass
                
            '''
            Save dataframe
            '''
            pd.set_option('io.hdf.default_format','table')
            df = pd.DataFrame(df, columns=columns)
            df = df.astype({'Nt':np.int32,'Nx':np.int32, 'mv':np.int32,
                            'other_costs':np.int32})
                    
            with pd.HDFStore('Experiment1.h5') as hdf:
                hdf.open()
                hdf.append(key='/'+integrator.name, value=df, format='table')
            print(time()-begin)
        ''' Remove duplicates '''
        with pd.HDFStore('Experiment1.h5') as hdf:
            for key in hdf.keys():
                hdf[key] = hdf[key].drop_duplicates()
                print('Key',key)
                print(hdf[key])
        
        print(time()-begin)