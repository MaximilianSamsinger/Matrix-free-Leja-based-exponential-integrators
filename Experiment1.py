from expleja import expleja
from AdvectionDiffusion1D import AdvectionDiffusion1D
import numpy as np
import pandas as pd
from Integrators import Integrator, rk2, rk4, cn2, exprb2
from time import time, sleep
from itertools import chain, product
from multiprocessing import Process, Lock

'''
Linear Case: dt u = adv dxx u + dif dx u
On the interval [0,1] with homogeneous Dirichlet boundary conditions,
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
    - Consider splitting this file into 2 files.
    - Adapt the code to handle the nonlinear case as well
'''

def compute_errors_and_costs(integrator, inputs, substeps, add_to_row): 
    ''' 
    Create dataframe 
    '''
    begin = time()
    data = [] # Will be filled with rows later on
    
    for target_error in integrator.target_errors:
        min_error, skip = 1, False # Indicates if computation can be stopped
        '''
        Set tolerances
        '''               
        if integrator.name == 'cn2':
            cn2.tol = target_error # rel. tolerance for gmres
        elif integrator.name == 'exprb2':
            exprb2.tol = [0,target_error,2,2] # rel. tolerance for expleja

        
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
                    if error < target_error: skip = True
                    if error < min_error: min_error = min(min_error, error)
                    elif skip and error > min_error*50: break
                else:
                    raise NameError('Method name not recognized, therefore it '
                                    + 'unclear when the computation finishes')
            except (FloatingPointError):
                pass
    print(integrator.columns)
    df = pd.DataFrame(data, columns=integrator.columns)
    df = df.astype({'substeps':np.int32,'Nx':np.int32, 'mv':np.int32,
                        'other_costs':np.int32})
    key = '/'+integrator.name # If we don't do that, we get an error.
    print(integrator.name, 'Nx:', add_to_row[0], 
          'time:', time()-begin, flush=True)
    return df, key

def solve_ODE(integrators, integrator_inputs, substeps, add_to_row, lock=None):
    
    for integrator in integrators:
        df, key = compute_errors_and_costs(integrator, integrator_inputs,
                                           substeps, add_to_row)
        ''' 
        Save dataframe 
        '''
        if lock is not None:
            lock.acquire() #For multiprocessing
        
        pd.set_option('io.hdf.default_format','table')
        with pd.HDFStore('Experiment1.h5') as hdf:
            hdf.open()
            hdf.append(key=key, value=df, format='table')
        
        if lock is not None:
            lock.release()

def Linear_Advection_Diffusion_Equation(Nx, adv, dif, Petype, lock):
    
    if Petype == 'pe':
        adv *= 2
        dif /= (Nx-1)
    elif Petype == 'Pe':
        pass
    else:
        raise TypeError("Petype needs to be either 'pe' or 'Pe'")
    
    t = 0 # Start time
    t_end = 1e-1 # Final time
    
    A, u = AdvectionDiffusion1D(Nx, adv, dif, periodic = False, 
                                h = None, asLinearOp = False)
    
    reference_solution = expleja(t_end, A, u)[0] # Double precision
    target_errors = [2**-10,2**-24]
    
    substeps = np.floor(1.12**np.array(range(1,151)))
    substeps = np.unique(substeps.astype('int'))  
    
    methods = [exprb2, cn2, rk2, rk4]
    columns = ['substeps', 'Nx', 'adv', 'dif', 'rel_error_2norm', 'mv', 
               'other_costs']
    
    integrators = [Integrator(method, reference_solution, target_errors, 
                              columns) for method in methods]
    
    integrator_inputs = [A, u, t, t_end]
    
    add_to_row = [Nx, adv, dif]
    
    solve_ODE(integrators, integrator_inputs, substeps, add_to_row, lock=lock)
    
if __name__ == '__main__':
    Nxlist = list(range(50,401,50))
    
    advs = [1e0]#, 1e-1, 1e-2] # Coefficient of advection matrix. Should be <= 1
    difs = [1e0, 1e-1, 1e-2] # Coefficient of diffusion matrix. Should be <= 1
    advdifs = list(chain(product([1e0], difs),product(advs, [1e0])))
    
    Petype = 'Pe'
    
    multiprocessing = False
    
    begin = time()
    if multiprocessing:
        lock = Lock()
        all_processes = [Process(target = Linear_Advection_Diffusion_Equation,
                                 args=(Nx, adv, dif, Petype, lock)
                                 ) for Nx in Nxlist for adv, dif in advdifs]
        
        for p in all_processes: p.start()
        for p in all_processes: p.join()
          
    else:
        for Nx, advdif in product(Nxlist, advdifs):
            adv, dif = advdif
            Linear_Advection_Diffusion_Equation(Nx, adv, dif, Petype, None)
            
    pd.set_option('io.hdf.default_format','table')
    with pd.HDFStore('Experiment1.h5') as hdf:
        for key in hdf.keys():
            hdf[key] = hdf[key].drop_duplicates()
            print('Key',key)
            print(hdf[key])
    print('Total time:', time()-begin)
    sleep(120)