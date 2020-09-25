import numpy as np
import pandas as pd
from Integrators import largestEV
from time import time

'''
For different fixed time stepsizes and matrix dimension we calculate the error
and the cost for each integration method
'''

#!ptrepack -o Experiment1.h5 Experiment11.h5

'''
Nx... discretize x-axis into Nx-1 parts
s... number of substeps used when integrating
'''

def Convert(Setting, AllSetting, name):
    '''
    Converts Settings to keyword arguments for the respective Integrators
    '''
    kwargs = {}
    if 'exprb' in name:
        if 'tol' in Setting:
            kwargs['tol'] = [0,Setting['tol'],2,2]
        if 'powerits' in Setting:
            if 'safetyfactor' in Setting:
                kwargs['normEstimator'] = (
                        largestEV,
                        {'powerits': Setting['powerits'],
                         'safetyfactor': Setting['safetyfactor']
                        }
                )
            else:
                kwargs['normEstimator'] = (
                        largestEV,
                        {'powerits': Setting['powerits']}
                )
        if 'dF' in AllSetting:
            kwargs['dF'] = AllSetting['dF']
    elif name == 'cn2':
        if 'tol' in Setting:
            kwargs['tol'] = Setting['tol']
        if 'dF' in AllSetting:
            kwargs['dF'] = AllSetting['dF']
    return kwargs

def compute_errors_and_costs(Integrator, Settings, add_to_row):
    '''
    Create dataframe
    '''
    begin = time()
    data = [] # Will be filled with rows later on

    for Setting in Settings[Integrator.name]:
        kwargs = Convert(Setting, Settings['all'], Integrator.name)

        if len(add_to_row) == 2:
            print(Integrator.name, 'Nx:', add_to_row[0],
                  'Pe:', 1./add_to_row[1], Setting)
        else:
            print(Integrator.name, 'Nx:', add_to_row[0],
                  'params:', add_to_row[1:], Setting)



        error = 1
        s_max = max(Settings['all']['substeps'])
        if Integrator.name in 'exprb' or Integrator.name == 'cn2':
            s_max = 10000
        s_break = s_max
        for s in Settings['all']['substeps']:
            try: # We skip computation when we encounter runtime errors
                '''
                Compute relative errors (2norm) and costs.
                We skip calculations when runtime warnings arise
                since the results would unusable anyways.
                '''
                with np.errstate(all='raise'): #Runtime warnings set to errors.
                    _, error, costs = Integrator.solve(s, **kwargs)
                    row = [s] + add_to_row + [error] + list(costs)
                    row += [value for key, value in Setting.items()]
                    print(Integrator.name, s, error)
                    data.append(row)

                '''
                If the error is small enough, skip further calculations
                '''
                if s > s_break:
                    break
                elif Integrator.name in ['rk2','rk4']:
                    if error < min(Settings["all"]["tol"]): break
                elif Integrator.name == 'cn2':
                    if error < Setting['tol'] and s_break == s_max:
                        s_break = 2*s
                elif 'exprb' in Integrator.name:
                    if Settings['all']['dF'] is not False:
                        # Otherwise we might mess up Experiment 1
                        if error < Setting['tol'] and s_break == s_max:
                            s_break = 5*s
                else:
                    raise NameError('Method name not recognized, therefore it '
                                    + 'unclear when the computation finishes')
            except (FloatingPointError, MemoryError):
                pass

    df = pd.DataFrame(data, columns=Integrator.columns)
    df = df.astype(Settings['all']['dftype'])
    key = '/'+Integrator.name # If we don't do that, we get an error.
    
    if len(add_to_row) == 2:
        print(Integrator.name, 'Nx:', add_to_row[0], 'Pe:',
              1./add_to_row[1], 'time:', time()-begin, flush=True)
    else:
        print(Integrator.name, 'Nx:', add_to_row[0], 'params:',
              add_to_row[1:], 'time:', time()-begin, flush=True)     
    return df, key

def solve_ODE(Integrators, Settings, add_to_row, filename, lock=None):

    for Integrator in Integrators:
        df, key = compute_errors_and_costs(Integrator, Settings, add_to_row)
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