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

t = 0 # Start time
t_end = 1e-1 # Final time

#!ptrepack -o Experiment1.h5 Experiment11.h5

if __name__ == '__main__':    
    ''' Runtime warnings kill performance and will not lead to results '''
    np.seterr(over='raise', invalid='raise') # Raise error instead of warning
    
    
    ''' 
    CONFIG
    '''
    '''
    Nx... discretize x-axis into Nx parts 
    Nt... Integrate with Nt substeps
    '''
    Nxlist = list(range(50,401,50))
    Ntlist = np.floor(1.12**np.array(range(70,79)))
    Ntlist = np.unique(Ntlist.astype('int'))
    
    #names = ['rk2', 'rk4', 'expeuler', 'crankn']
    #methods = [rk2_matrixinput, rk4_matrixinput, expeuler, crankn]
    names = ['crankn']
    methods = [crankn]
    crankn.parameter_name = 'gmres_tol'
    
    for tol in [2**-k for k in range(23,33)]:
        crankn.parameter = tol
        
        
        assert(len(names) == len(methods))
    
        
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
                columns = ['Nt', 'Nx', 'Pe', 'error', 'mv', 'other_costs', 
                           'adv_coeff', 'dif_coeff']
                if hasattr(integrator.method,'parameter'):
                    columns.append(integrator.method.parameter_name)
                            
                for Nt in Ntlist:
                    try:
                        ''' Calculate error and costs'''
                        _, error, functionEvaluations, otherCosts = \
                                            integrator.solve(A, t, u, t_end, Nt)
                        
                        ''' Store relevant arguments and parameters '''
                        tmp = [Nt, Nx, Pe, error, functionEvaluations, 
                               otherCosts, adv_coeff, dif_coeff]
                        if hasattr(integrator.method,'parameter'):
                            tmp.append(integrator.method.parameter)
                        
                        df.append(tmp)
                        
                    except (FloatingPointError):
                        pass
                
                
                ''' Save dataframe '''
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