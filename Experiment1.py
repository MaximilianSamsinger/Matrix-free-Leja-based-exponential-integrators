from expleja import expleja
from AdvectionDiffusion1D import AdvectionDiffusion1D
import numpy as np
import pandas as pd
from Integrators import MatrixIntegrator, rk2_matrixinput, rk4_matrixinput, \
                        cn2, exprk2
from time import time, sleep
from multiprocessing import Process, Lock

'''
Linear Case: dt u = adv dxx u + dif dx u
On the interval [0,1] with dirichlet homogeneous Dirichlet boundary conditions,
for t in [0,0.1]

For different fixed time stepsizes and matrix dimension we calculate the error
and the cost for each integration method
'''

#!ptrepack -o Experiment1.h5 Experiment11.h5

'''
The costs of the crank-nicolson method largly depend on the tolerance chosen 
for gmres.
We add an extra column to the dataframe in Experiment1.h5 which specifies the
chosen tolerance for gmres.
'''

''' 
Nx... discretize x-axis into Nx parts 
Nt... Integrate with Nt substeps
'''

'''
TODO:
    - Consider splitting this file into 2 files.
    - Run Experiment with Pe fixed.
    - Adapt the code to handle the nonlinear case as well
'''

Nxlist = list(range(50,401,25))
Ntlist = np.floor(1.12**np.array(range(1,121)))
Ntlist = np.unique(Ntlist.astype('int'))

t = 0 # Start time
t_end = 1e-1 # Final time

adv = 1e0 # Multiply advection matrix with adv. Should be <= 1
dif = 1e0 # Multiply diffusion matrix with dif. Should be <= 1

names = ['exprk2', 'cn2', 'rk2', 'rk4']
methods = [exprk2, cn2, rk2_matrixinput, rk4_matrixinput] 

target_errors = [2**-10,2**-24]
columns = ['Nt', 'Nx', 'adv', 'dif', 
          'abs_error_2', 'rel_error_2', 'abs_error_inf', 'rel_error_inf',
          'mv', 'other_costs']

def compute_errors_and_costs(integrator, A, u, Nx, Ntlist = Ntlist, 
                             t = t, t_end = t_end, adv = adv, dif = dif): 
    ''' 
    Runtime warnings due to overflow lead to unusable results. We skip them.
    '''
    #np.seterr(over='raise', invalid='raise') # Raise error instead of warning              
    #np.seterr(all='raise')
    ''' 
    Create dataframe 
    '''
    begin = time()
    data = []
    for target_error in integrator.target_errors:
        if integrator.name == 'cn2':
            cn2.gmrestol = target_error/500 # Relative tolerance used by gmres
        elif integrator.name == 'exprk2':
            exprk2.tol = [0,target_error,2,2] # Tolerance specified for expleja
          
        skip = False
        for Nt in Ntlist:               
            try: # We skip computation when we encounter overflow errors 
                ''' 
                Compute errors and costs
                '''
                with np.errstate(all='raise'):
                    _, errors, costs = integrator.solve(A, t, u, t_end, Nt)
                row = [Nt, Nx, adv, dif] + list(errors) + list(costs)
                if integrator.name in ['cn2','exprk2']:
                    row += [target_error]
                data.append(row)
                    
                
                ''' 
                If the error is small enough, skip further calculations
                '''
                if integrator.name in ['rk2','rk4']:
                    if errors[1] < target_error:
                        break
                
                elif integrator.name == 'cn2':
                    if errors[1] < target_error:
                        while errors[1] < target_error:
                            cn2.gmrestol *= 1.2
                            _, errors, costs = integrator.solve(
                                    A, t, u, t_end, Nt)
                            data.append([Nt, Nx, adv, dif] 
                                + list(errors) + list(costs)
                                + [target_error])
                            if cn2.gmrestol >= target_error:
                                break
                        break
                
                elif integrator.name == 'exprk2':
                    if errors[1] < target_error/50:
                        break
                    elif skip and errors[1] > target_error*50:
                        break
                    elif errors[1] < target_error:
                        skip = True
            except (FloatingPointError):
                pass
    df = pd.DataFrame(data, columns=integrator.columns)
    df = df.astype({'Nt':np.int32,'Nx':np.int32, 'mv':np.int32,
                        'other_costs':np.int32})
    key = '/'+integrator.name
    print(integrator.name, 'Nx:', Nx, 'time:', time()-begin, flush=True)
    return df, key

def solve_advection_diffusion_equation(Nx, lock=None, 
                                       t = t, t_end = t_end, 
                                       adv = adv, dif = dif,
                                       columns = columns,
                                       names = names, methods = methods,
                                       target_errors = target_errors):
    
    A, u = AdvectionDiffusion1D(Nx, adv, dif, periodic = False, 
                                h = None, asLinearOp = False)
    
    reference_solution = expleja(t_end, A, u)[0] # Double precision
    
    integrators = [MatrixIntegrator(name, method, reference_solution
                             ) for name, method in zip(names, methods)]
    
    for integrator in integrators:
        if integrator.name in ['rk2','rk4']:
            integrator.target_errors = [min(target_errors)]
            integrator.columns = columns
        elif integrator.name in ['cn2','exprk2']:
            integrator.target_errors = target_errors
            integrator.columns = columns + ['target_error']
        
        df, key = compute_errors_and_costs(integrator, A, u, Nx)
        ''' 
        Save dataframe 
        '''
        if lock is not None:
            lock.acquire()
        
        pd.set_option('io.hdf.default_format','table')
        with pd.HDFStore('Experiment1.h5') as hdf:
            hdf.open()
            hdf.append(key=key, value=df, format='table')
        
        if lock is not None:
            lock.release()
        
if __name__ == '__main__':
    begin = time()
    multiprocessing = True
    start = time()
    if multiprocessing:
        lock = Lock()
        all_processes = [Process(target=solve_advection_diffusion_equation, 
                    args=(Nx,lock)) for Nx in Nxlist]
        
        for p in all_processes:
            p.start()

        for p in all_processes:
          p.join()
          
    else:
        for Nx in Nxlist:
            solve_advection_diffusion_equation(Nx)
            
    pd.set_option('io.hdf.default_format','table')
    with pd.HDFStore('Experiment1.h5') as hdf:
        for key in hdf.keys():
            hdf[key] = hdf[key].drop_duplicates()
            print('Key',key)
            print(hdf[key])
    print('Total time:', time()-begin)
    sleep(30)