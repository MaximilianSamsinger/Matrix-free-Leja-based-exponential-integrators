import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from itertools import chain, product
from datapreperation import IntegratorData, get_optimal_data

'''
DISCRETIZED ONE DIMENSIONAL LINEAR ADVECTION-DIFFUSION EQUATION

Experiment 2:
    Here we fix the highest acceptable error (2-norm) for the solution
    of the linear advection-diffusion equantion and look at all datapoints
    for which an integrator satisfies:
        - The maximal integration error is small enough
        - The costs (matrix-vector multiplication) are minimal

    The integrators are
        - rk2... Runge-Kutta method of order 2
        - rk4... Runge-Kutta method of order 4
        - cn2... Crank-Nicolson method of order 2
        - exprb2... Exponential euler method of order 2, in this case EXACT

Note: We write exprb2, even though we only compute the matrix exponential
    function of the discretized matrix using expleja with half/single precision
    and fixed step size.
'''

'''
Global plot parameters
'''
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rcParams['lines.linewidth'] = 3

'''
Load data
'''
filelocation = 'HDF5-Files' + os.sep + 'Experiment2.h5'
with pd.HDFStore(filelocation) as hdf:
    keys = hdf.keys()
    Integrators = {key:IntegratorData(filelocation,key) for key in keys}

pd.set_option('display.max_columns', 500) # Easier to investigate data  
pd.set_option('display.width', 1000)  # Easier to investigate data  
'''
CONFIG
'''
maxerror = str(2**-24)

if maxerror == str(2**-24):
    precision = '$2^{-24}$'
    precision_type = 'single'

elif maxerror == str(2**-10):
    precision = '$2^{-10}$'
    precision_type = 'half'

errortype = 'relerror'

save = False # Flag: If True, figures will be saved as pdf
save_path = 'figures' + os.sep + 'Experiment2' + os.sep

params = [[α, β, γ] for α in [0.1, 0.01] for β in [1, 0.1, 0.01] for γ in [1]]


def savefig(number, save=False, add_to_filename = None):
    if add_to_filename is None:
        filename = f'{number}, {precision_type}.pdf'
    else:
        filename = f'{number}, {precision_type}, {add_to_filename}.pdf'
    if save:
        plt.savefig(save_path + filename
                    , format='pdf', bbox_inches='tight', transparent=True)
        print('File saved')
        plt.close()


for param in params:
    print(param)
    assert(param[0] <= 1)
    
    adv = param[1]

    suptitle = f'α={param[0]}, β={param[1]}'
    paramtext = f'α={param[0]}, β={param[1]}'


    for key in keys:
        get_optimal_data(Integrators[key], float(maxerror), errortype, param)


    
    '''
    1.3 Plot matrix dimension (Nx) vs costs/substep (mv)
    '''
    fig, ax = plt.subplots()
    fig.suptitle(suptitle)
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','relerror', label=key[1:], ax=ax)
    ax.set_title('Minimal costs for ' + precision + ' results')
    ax.set_xlabel('N')


    savefig(3, save, paramtext)

