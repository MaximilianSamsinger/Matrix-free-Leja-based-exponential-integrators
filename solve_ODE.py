import numpy as np
import pandas as pd
from Integrators import Integrator, rk2, rk4, cn2, exprb2
from time import time
from itertools import product

'''
For different fixed time stepsizes and matrix dimension we calculate the error
and the cost for each integration method
'''

#!ptrepack -o Experiment1.h5 Experiment11.h5

''' 
Nx... discretize x-axis into Nx-1 parts 
s... number of substeps used when integrating
'''

def compute_errors_and_costs(integrator, inputs, substeps, add_to_row): 
    ''' 
    Create dataframe 
    '''
    begin = time()
    data = [] # Will be filled with rows later on
    
    target_errors = integrator.target_errors
    powerits = integrator.powerits
    
    for target_error, powerit in product(target_errors, powerits):
        min_error, skip = 1, False # Indicates if computation can be stopped
        '''
        Set tolerances
        '''               
        if integrator.name == 'cn2':
            cn2.tol = target_error # rel. tolerance for gmres
        elif integrator.name == 'exprb2':
            exprb2.tol = [0,target_error,2,2] # rel. tolerance for expleja
            exprb2.powerits = powerit
        
        print(integrator.name, target_error, powerit, 'Nx:', add_to_row[0], 
              'Pe:', 1./add_to_row[1])
        
        for s in substeps:            
            try: # We skip computation when we encounter runtime errors 
                ''' 
                Compute relative errors (2norm) and costs. 
                We skip calculations when runtime warnings arise 
                since the results would unusable anyways.
                '''
                with np.errstate(all='raise'): #Runtime warnings set to errors.
                    _, error, costs = integrator.solve(*inputs, s)
                    row = [s] + add_to_row + [error] + list(costs) 
                if integrator.name in ['cn2','exprb2']: 
                    row += [target_error]
                data.append(row)
                    
                ''' 
                If the error is small enough, skip further calculations
                '''
                if integrator.name in ['rk2','rk4','cn2']:
                    if error < target_error: break
                elif integrator.name == 'exprb2':
                    if exprb2.maxsubsteps < s: break
                    if error < target_error: skip = True
                    if error < min_error: min_error = min(min_error, error)
                    elif skip and error > min_error*50: break
                else:
                    raise NameError('Method name not recognized, therefore it '
                                    + 'unclear when the computation finishes')
            except (FloatingPointError):
                pass
    df = pd.DataFrame(data, columns=integrator.columns)
    df = df.astype({'substeps':np.int32,'Nx':np.int32, 'mv':np.int32,
                        'misc':np.int32})
    key = '/'+integrator.name # If we don't do that, we get an error.
    print(integrator.name, 'Nx:', add_to_row[0], 'Pe:', 
          1./add_to_row[1], 'time:', time()-begin, flush=True)
    return df, key

def solve_ODE(integrators, integrator_inputs, substeps, add_to_row, filename,
              lock=None):
    
    for integrator in integrators:
        df, key = compute_errors_and_costs(integrator, integrator_inputs,
                                           substeps, add_to_row)
        ''' 
        Save dataframe 
        '''
        if lock is not None:
            lock.acquire() #For multiprocessing
        
        pd.set_option('io.hdf.default_format','table')
        with pd.HDFStore(filename) as hdf:
            hdf.open()
            hdf.append(key=key, value=df, format='table')
        
        if lock is not None:
            lock.release()