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

Experiment 1:
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
CONFIG
'''
maxerror = str(2**-10)
save = False # Flag: If True, figures will be saved as pdf
save_path = 'figures' + os.sep + 'Experiment1' + os.sep
adv = 1.0 # Coefficient of advection matrix. Do not change.
difs = [1e0, 1e-1, 1e-2] # Coefficient of diffusion matrix. Should be <= 1

'''
Load data
'''
filelocation = 'HDF5-Files' + os.sep + 'Experiment1.h5'
with pd.HDFStore(filelocation) as hdf:
    keys = hdf.keys()
    Integrators = {key:IntegratorData(filelocation,key) for key in keys}
    

'''
For saving and plotting
'''
if maxerror == str(2**-24):
    precision = '$2^{-24}$'
    precision_type = 'single'
elif maxerror == str(2**-10):
    precision = '$2^{-10}$'
    precision_type = 'half'

errortype = 'relerror'

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



for dif in difs:
    print(dif)
    assert(dif <= 1)

    for key in keys:
        get_optimal_data(Integrators[key], float(maxerror), errortype, dif)

    suptitle = f'Pe = {adv/dif}'
    Petext = f' for Pe = {adv/dif}'


    '''
    1.1 Plot matrix dimension (Nx) vs matrix-vector multiplications (mv)
    '''
    fig, ax = plt.subplots()
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','mv', label=key[1:], ax=ax)
    ax.set_title(f'Achieving error ≤ {precision} {Petext}')
    ''' Optimal in the sense of cost minimizing '''
    ax.set_xlabel('N')
    ax.set_ylabel('Matrix-vector multiplications')
    #ax.set_ylim([5e2,1.5e5])
    ax.set_yscale('log')

    savefig(1, save, f'Pe={adv/dif}')
    
    precisiontext = '\n achieving error ≤ ' + precision
    title = 'Optimal time step ' + precisiontext + Petext
    
    '''
    1.2 Plot matrix dimension (Nx) vs optimal time step size (tau)
    '''
    fig, ax = plt.subplots()
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','tau', label=key[1:], ax=ax)


    CFLA = df.gridsize/adv
    CFLD = 0.5*df.gridsize**2/df.dif
    ax.plot(df.Nx, CFLA, label="$CFL_{adv}$",linestyle='-.')
    ax.plot(df.Nx, CFLD, label="$CFL_{dif}$",linestyle=':')

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('Optimal time step')
    ax.set_yscale('log')

    savefig(2, save, f'Pe={adv/dif}')

    '''
    1.3 Plot matrix dimension (Nx) vs matrix-vector multiplications (mv)
    '''
    fig, ax = plt.subplots()
    fig.suptitle(suptitle)
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','m', label=key[1:], ax=ax)
    ax.set_title(f'Minimal costs for {precision} results')
    ax.set_xlabel('N')
    ax.set_ylabel('Matrix-vector multiplications per timestep')
    ax.set_ylim([0,120])

    savefig(3, save, f'Pe={adv/dif}')
    
    '''
    Experiment 1.5 (Bonus):
    
    Compute the error of the expleja method for varying matrix dimensions NxN.
    Normally this would result would be almost exact, but in this case we fix the
    number of substeps and assume single precision.
    '''
    
    fig, ax = plt.subplots(1, 1, sharex=True)
    fig.suptitle(suptitle + '\n Single precision expleja')
    
    data = Integrators['/exprb2'].data
    data = data.loc[(data['dif'] == dif)
                    & (data[errortype] <= 10)
                    & (data['tol'] == float(maxerror))]
    data = data.sort_values(by='substeps')
    
    for label, df in data.groupby('Nx'):
        if label%100 == 0:
            df.plot('substeps',errortype, ax=ax, label='N = ' + str(label))
    
    ax.set_xlabel('Number of substeps')
    ax.set_ylabel('Relative error')
    ax.set_xscale('log')
    ax.set_yscale('log')    
    
    savefig(5, save, f'Pe={adv/dif}')


'''
1.4 Plot Matrix-vector multiplications (Nx) vs optimal time step size (tau)
'''
fig, ax = plt.subplots()
linestyles = ['solid', 'dotted', 'dashed']

for k, dif in enumerate([1e0,1e-1,1e-2]):
    plt.gca().set_prop_cycle(None)
    for key in keys:
        get_optimal_data(Integrators[key], float(maxerror), errortype, dif)
        df = Integrators[key].optimaldata

        label = key[1:] if k == 0 else '_nolegend_'
        df.plot('mv','tau', label=label, ax=ax, linestyle = linestyles[k])

        for i,j in df.Nx.items():
            if j in [df.Nx.iloc[k] for k in [0,-1]]:
                ax.annotate('N=' + str(j), xy=(df.mv[i], df.tau[i]))

for k, l in enumerate(linestyles):
    ax.plot([],[], 'k', linestyle = l,
            label = 'Pe = ' + str((difs[k]/dif)))

ax.set_title(title)
ax.legend()
ax.set_xlabel('Matrix-vector multiplications')
ax.set_ylabel('Optimal time step')
ax.set_xscale('log')
ax.set_yscale('log')

savefig(4, save)



'''
1.6 Plot Peclet number (Nx) vs something else
'''
'''
fig, ax = plt.subplots()
linestyles = ['solid', 'dotted', 'dashed']
difs = [1e0,1e-1,1e-2]
for k, dif in enumerate([1e0,1e-1,1e-2]):
    plt.gca().set_prop_cycle(None)
    for key in keys:
        get_optimal_data(Integrators[key], float(maxerror), errortype, dif)
        df = Integrators[key].optimaldata

        if k == 0:
            df.plot('pe','mv', label=key[1:], ax=ax,
                    linestyle = linestyles[k])
        else:
            df.plot('pe','mv', label='_nolegend_', ax=ax,
                    linestyle = linestyles[k])
        for i,j in df.Nx.items():
            if j in [df.Nx.iloc[k] for k in [0,-1]]:
                ax.annotate('N=' + str(j), xy=(df.mv[i], df.tau[i]))

for k, l in enumerate(linestyles):
    ax.plot([],[], 'k', linestyle = l, label='pe = ' + str(int(adv/difs[k])))

ax.set_title(title)
ax.legend()
ax.set_xlabel('Grid peclet number')
ax.set_ylabel('Matrix-vector multiplications')
ax.set_xscale('log')
ax.set_yscale('log')

savefig(6, save)
'''
