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
keys = [keys[k] for k in [0,1,4,5,2,3]]

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
    1.1 Plot matrix dimension (Nx) vs matrix-vector multiplications (m)
    '''
    fig, ax = plt.subplots()
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','cost', label=key[1:], ax=ax)
    ax.set_title(f'Achieving error ≤ {precision}, {paramtext}')
    ''' Optimal in the sense of cost minimizing '''
    ax.set_xlabel('N')
    ax.set_ylabel('Cost')
    if precision_type == 'half':
        ax.set_ylim([1e0,1.1e5])
    else:
        ax.set_ylim([1e0,1.1e5])
    ax.set_yscale('log')

    savefig(1, save, paramtext)
    
    precisiontext = f'\n achieving error ≤ {precision}'
    title = f'Optimal time step {precisiontext}, {paramtext}'
    '''
    1.2 Plot matrix dimension (Nx) vs optimal time step size (tau)
    '''
    fig, ax = plt.subplots()
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','tau', label=key[1:], ax=ax)


    CFLA = df.gridsize/df.β
    CFLD = 0.25*df.gridsize**2/df.α # Yes, 0.25 is correct
    ax.plot(df.Nx, CFLA, label="$CFL_{adv}$",linestyle='-.')
    ax.plot(df.Nx, CFLD, label="$CFL_{dif}$",linestyle=':')

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('Optimal time step')
    ax.set_yscale('log')

    savefig(2, save, paramtext)

    '''
    1.3 Plot matrix dimension (Nx) vs costs/substep (mv)
    '''
    fig, ax = plt.subplots()
    fig.suptitle(suptitle)
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','m', label=key[1:], ax=ax)
    ax.set_title('Minimal costs for ' + precision + ' results')
    ax.set_xlabel('N')
    ax.set_ylabel('Costs per timestep')
    ax.set_ylim([0,100])

    savefig(3, save, paramtext)
    
    '''
    1.4 Plot Matrix-vector multiplications (Nx) vs optimal time step size (tau)
    '''
    fig, ax = plt.subplots()
    
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('cost','tau', label=key[1:], ax=ax)
    
        for i,j in df.Nx.items():
            if j in [df.Nx.iloc[k] for k in [0,-1]]:
                ax.annotate('N=' + str(j), xy=(df.cost[i], df.tau[i]))
    
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('Matrix-vector multiplications')
    ax.set_ylabel('Optimal time step')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim([1e1,1e5])
    ax.set_ylim([1e-5,1e-1])
    
    savefig(4, save, paramtext)

