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
The costs of the crank-nicolson method largly depend on the tolerance chosen 
for gmres.
We add an extra column to the dataframe in Experiment1.h5 which specifies the
chosen tolerance for gmres.
'''

#!ptrepack -o Experiment1.h5 Experiment11.h5

if __name__ == '__main__':
    hdf = pd.HDFStore('Experiment1.h5')
    
    ''' Runtime warnings kill performance and will not lead to results '''
    np.seterr(over='raise', invalid='raise') # Raise error instead of warning
    
    t = 0 # Start time
    t_end = 1e-1 # Final time
    
    ''' 
    Nx... discretize x-axis into Nx parts 
    Nt... Integrate with Nt substeps
    '''
    Nxlist = list(range(50,401,50))
    Ntlist = np.floor(1.12**np.array(range(1,81)))
    Ntlist = np.unique(Ntlist.astype('int'))
    
    names = ['rk2', 'rk4', 'expeuler']
    
    methods = [rk2_matrixinput, rk4_matrixinput, expeuler] 
    
    columns = ['Nt', 'Nx', 'Pe', 'error', 'mv', 
               'other_costs', 'adv_coeff', 'dif_coeff', 'gmres_tol']
    #df = pd.DataFrame(columns=columns)
    
    for Nx in Nxlist:
        print('Nx:', Nx)
        adv_coeff = 1e0 # Multiply advection matrix with adv_coeff
        dif_coeff = 1e0 # Multiply diffusion matrix with dif_coeff
        Pe = adv_coeff/dif_coeff
        A, u = AdvectionDiffusion1D(Nx, adv_coeff, dif_coeff, periodic = False, 
                                    h = None, asLinearOp = False)
        
        exact_solution = expleja(t_end, A, u)[0] #Note: double precision

        
        integrators = [MatrixIntegrator(name, method, Ntlist, exact_solution
                                 ) for name, method in zip(names, methods)]
        
        begin = time()
        for integrator in integrators:
            ''' Create and fill pandas dataframe '''
            df = []
            print(integrator.name)
            for Nt in Ntlist:
                ''' Calculate error and costs'''
                
                try:
                    _, error, functionEvaluations, otherCosts = \
                                        integrator.solve(A, t, u, t_end, Nt)
                    df.append([Nt, Nx, Pe, error, functionEvaluations, 
                               otherCosts, adv_coeff, dif_coeff])
                except (FloatingPointError):
                    pass
                
            
            ''' Save dataframe '''
            df = pd.DataFrame(df, columns=columns)
            df = df.astype({'Nt':np.int32,'Nx':np.int32, 'mv':np.int32
                              , 'other_costs':np.int32})
            df.to_hdf('Experiment1.h5', key='/'+integrator.name,
                      format='table')
            
            ''' Remove duplicates '''
            try:
                hdf[integrator.name] = hdf[integrator.name].drop_duplicates()
            except KeyError:
                ''' Prevents an error, if the HDF file didn't exist before '''
                hdf = pd.HDFStore('Experiment1.h5')
                hdf[integrator.name] = hdf[integrator.name].drop_duplicates()
        print(time()-begin)
        
    #print(df[df.columns.difference(['adv_coeff', 'dif_coeff','other_costs'])])
    
    '''
    exp = df[df['Integrator'] == 'ExponentialEuler']
    exp.plot(x = 'Nt', y = 'error', logy = True)
    tollist = [10**-k for k in range(4,9)]
    '''