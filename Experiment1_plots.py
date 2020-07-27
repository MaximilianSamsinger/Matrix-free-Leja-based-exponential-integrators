import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from itertools import chain, product
from datapreperation import IntegratorData, get_optimal_data, \
    global_plot_parameters, savefigure
import matplotlib as mlp

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
CONFIG
'''
LEGEND_SIZE = 8
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

maxerror = str(2**-10)

save = True # Flag: If True, figures will be saved as pdf
save_path = 'Figures' + os.sep + 'Experiment1' + os.sep

adv = 1.0 # Coefficient of advection matrix. Do not change.
difs = [1e0, 1e-1, 1e-2] # Coefficient of diffusion matrix. Should be <= 1

'''
Global plot parameters
'''
scaling = 0.9
width = 426.79135/72.27
golden_ratio = (5**.5 - 1) / 2
fig_dim = lambda fraction: (width*fraction, width*fraction*golden_ratio)

global_plot_parameters(SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,fig_dim(scaling),
                       LEGEND_SIZE)
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

savefig = lambda num, save, *args : savefigure(save_path, num, save, *args)

for dif in difs:
    print(dif)
    assert(dif <= 1)

    ''' Optimal in the sense of cost minimizing '''
    for key in keys:
        get_optimal_data(Integrators[key], float(maxerror), errortype, dif)

    paramtext = '{{$\\mathrm{Pe}$}}=' + f'{adv/dif}'
    titletext = f'{{{precision_type.capitalize()}}} precision, ' \
        + paramtext
    savetext = f'Pe={adv/dif}'

    '''
    1.1 Plot matrix dimension (Nx) vs matrix-vector multiplications (mv)
    '''
    fig, ax = plt.subplots()
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','mv', label=key[1:], ax=ax)
    ax.set_title(titletext)
    ax.set_xlabel('{{$N$}}')
    ax.set_ylabel('Matrix-vector multiplications')
    ax.set_yscale('log')

    savefig(1, save, precision_type, savetext)
    
    '''
    1.2 Plot matrix dimension (Nx) vs optimal time step size (tau)
    '''
    fig, ax = plt.subplots()
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','tau', label=key[1:], ax=ax)


    #CFLA = df.gridsize/adv
    CFLD = 0.5*df.gridsize**2/df.dif
    #ax.plot(df.Nx, CFLA, label="$C_{adv}$",linestyle='-.')
    ax.plot(df.Nx, CFLD, label="$C_{dif}$",linestyle=':')

    ax.set_title(titletext)
    ax.legend()
    ax.set_xlabel('{{$N$}}')
    ax.set_ylabel('Optimal time step')
    ax.set_yscale('log')

    savefig(2, save, precision_type, savetext)

    '''
    1.3 Plot matrix dimension (Nx) vs matrix-vector multiplications (mv)
    '''
    fig, ax = plt.subplots()
    #fig.suptitle(titletext)
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('Nx','m', label=key[1:], ax=ax)
    ax.set_title(titletext)
    ax.set_xlabel('{{$N$}}')
    ax.set_ylabel('Matrix-vector multiplications per timestep')
    ax.set_ylim([0,120])

    savefig(3, save, precision_type, savetext)
    

    '''
    1.4 Plot Matrix-vector multiplications (Nx) vs optimal time step size (tau)
    '''
    fig, ax = plt.subplots()
    linestyles = ['solid', 'dotted', 'dashed']
    
    
    for key in keys:
        df = Integrators[key].optimaldata
        df.plot('mv','tau', label=key[1:], ax=ax)

        for i,j in df.Nx.items():
            if j in [df.Nx.iloc[k] for k in [0,-1]]:
                ax.annotate('N=' + str(j), xy=(df.mv[i], df.tau[i]))
    
    ax.set_title(titletext)
    ax.legend()
    ax.set_xlabel('Matrix-vector multiplications')
    ax.set_ylabel('Optimal time step')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    savefig(4, save, precision_type, savetext)


    '''
    Experiment 1.5:
    
    Compute the error of the expleja method for varying matrix dimensions NxN.
    Normally this would result would be almost exact, but in this case we fix the
    number of substeps and assume single precision.
    '''
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=fig_dim(0.75))
 
    fig.suptitle(f'{{{precision_type.capitalize()}}} precision expleja, '
                 + paramtext)
    
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
    if dif == 1e-1:
        ax.set_xlim([0,5e2])
    savefig(5, save, precision_type, savetext)


    '''
    1. Multi plot of 1.1 and 1.2
    '''

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row')
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(paramtext)
    
    for k, error in enumerate([2**-10,2**-24]):
        with pd.HDFStore(filelocation) as hdf:
            keys = hdf.keys()
            Integrators = {key:IntegratorData(filelocation,key) for key in keys}
        for key in keys:
            get_optimal_data(Integrators[key], float(error), errortype, dif)
        
        ax = axes[0+k]
        for key in keys:
            df = Integrators[key].optimaldata
            df.plot('Nx','mv', label=key[1:], ax=axes[0+k])
            
        ax.set_xlabel('{{$N$}}')
        ax.set_yscale('log')
        if k == 0:
            ax.set_title('Half precision')
            ax.set_ylabel('Matrix-vector multiplications')
            ax.get_legend().remove()
        else:
            ax.set_title('Single precision')
            ax.legend()
        ax = axes[2+k]
        
        for key in keys:
            df = Integrators[key].optimaldata
            df.plot('Nx','tau', label=key[1:], ax=axes[2+k])
        #ax.plot(df.Nx, CFLA, label="$CFL_{adv}$",linestyle='-.')
        ax.plot(df.Nx, CFLD, label="$CFL_{dif}$",linestyle=':')
        ax.set_xlabel('{{$N$}}')
        if k == 0:
            ax.set_ylabel('Optimal time step')
            
        ax.get_legend().remove()
        ax.set_yscale('log')
    
    fig.align_ylabels()
    fig.subplots_adjust(top=0.93)
    
    savefig('multi', save, savetext)

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