import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from datapreperation import IntegratorData, get_optimal_data, \
    global_plot_parameters, savefigure

'''
DISCRETIZED ONE/TWO DIMENSIONAL NONLINEAR ADVECTION-DIFFUSION EQUATION:
    Here we fix the highest acceptable error (2-norm) for the solution
    of the linear advection-diffusion equantion and look at all datapoints
    for which an integrator satisfies:
        - The maximal integration error is small enough
        - The costs (function evaluations) are minimal
    
    The integrators are
        - rk2... Runge-Kutta method of order 2
        - rk4... Runge-Kutta method of order 4
        - cn2... Crank-Nicolson method of order 2
        - exprb2... Exponential euler method of order 2, in this case EXACT
'''

'''
CONFIG
'''

LEGEND_SIZE = 8
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

maxerror = str(2**-24)

filename = 'Experiment4'
filelocation = 'HDF5-Files' + os.sep + filename + '.h5'
save = False # Flag: If True, figures will be saved (as pdf)
save_path = 'Figures' + os.sep + filename + os.sep

params = [[α, β, γ] for α in [0] for β in [1] for γ in [0]]

'''
Global plot parameters
'''
scaling = 0.9
width = 426.79135/72.27
golden_ratio = (5**.5 - 1) / 2
fig_dim = lambda fraction: (width*fraction, width*fraction*golden_ratio)

global_plot_parameters(SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,fig_dim(scaling),
                       LEGEND_SIZE)

default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    default_colors[k] for k in [2,1,9,0,5]
])

'''
Load data
'''
with pd.HDFStore(filelocation) as hdf:
    keys = hdf.keys()
    try:
        keys.remove('/exprb3')
    except ValueError:
        pass
    Integrators = {key:IntegratorData(filelocation,key) for key in keys}
#keys = [keys[k] for k in [0,1,4,5,2,3]]
    
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

pd.set_option('display.max_columns', 500) # Easier to investigate data  
pd.set_option('display.width', 1000)  # Easier to investigate data  

savefig = lambda num, save, *args : savefigure(save_path, num, save, *args)

for param in params:
    print(param)
    assert(param[0] <= 1)
    adv = param[1]
    
    ''' Optimal in the sense of cost minimizing '''
    for key in keys:
        get_optimal_data(Integrators[key], float(maxerror), errortype, param)

    paramtext = '{{$\\alpha$}}='+f'{param[0]}'+', {{$\\beta$}}='+f'{param[1]}'
    titletext = f'{{{precision_type.capitalize()}}} precision, ' \
        + paramtext
    savetext = f'α={param[0]}, β={param[1]}'
    
    '''
    1.1 Plot matrix dimension (Nx) vs matrix-vector multiplications (m)
    '''
    #fig, ax = plt.subplots()
    for key in keys:
        df = Integrators[key].optimaldata
        #df.plot('Nx','cost', label=key[1:], ax=ax)
    
    # ax.set_title(titletext)
    # ax.set_xlabel('{{$N$}}')
    # ax.set_ylabel('Megabytes read or written')
    # ax.set_yscale('log')

    # savefig(1, save, precision_type, savetext)
    
    # '''
    # 1.2 Plot matrix dimension (Nx) vs optimal time step size (tau)
    # '''
    # fig, ax = plt.subplots()
    # for key in keys:
    #     df = Integrators[key].optimaldata
    #     df.plot('Nx','tau', label=key[1:], ax=ax)

    
    
    
    
# =============================================================================
#     if isproblem2D:
#         CFLD = 0.25*df.gridsize**2/df.α 
#         CFLA = 0.5*df.gridsize/df.β
#     else:
# =============================================================================
    CFLD = 0.5*df.gridsize**2/df.α
    CFLA = df.gridsize/df.β
    CFL = pd.concat([CFLD, CFLA], axis=1).min(axis=1)
    
    if param[0] == 1:
        theoreticalmv = df.α*.8/df.gridsize**2
        suptitle = "Heat equation"
    elif param[1] == 1:
        theoreticalmv = df.β*.4/df.gridsize
        suptitle = "Advection equation"
    else:
        raise NotImplementedError
    
    # #ax.plot(df.Nx, CFLA, label="$C_{adv}$",linestyle='-.')
    # ax.plot(df.Nx, CFLD, label="$C_{dif}$",linestyle=':')

    # ax.set_title(titletext)
    # ax.legend()
    # ax.set_xlabel('{{$N$}}')
    # ax.set_ylabel('Optimal time step')
    # ax.set_yscale('log')

    # savefig(2, save, precision_type, savetext)

    # '''
    # 1.4 Plot cost (cost) vs optimal time step size (tau)
    # '''
    
    # fig, ax = plt.subplots()
    
    # for key in keys:
    #     df = Integrators[key].optimaldata
    #     df.plot('cost','tau', label=key[1:], ax=ax)
    
    #     for i,j in df.Nx.items():
    #         if j in [df.Nx.iloc[k] for k in [0,-1]]:
    #             ax.annotate('N=' + str(j), xy=(df.cost[i], df.tau[i]))
    
    # ax.set_title(titletext)
    # ax.legend()
    # ax.set_xlabel('Megabytes read or written')
    # ax.set_ylabel('Optimal time step')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    # #ax.set_xlim([1e1,1e5])
    # #ax.set_ylim([1e-5,1e-1])
    
    # savefig(4, save, precision_type, savetext)

    # '''
    # 1.3 Plot matrix dimension (Nx) vs Jacobian-vector products (mv)
    # '''
    
    # fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey='row')
    # axes = axes.flatten()
    # fig.subplots_adjust(hspace=0, wspace=0)
    # fig.suptitle(paramtext)
    
    # for k, error in enumerate([2**-10,2**-24]):
    #     with pd.HDFStore(filelocation) as hdf:
    #         keys = hdf.keys()
    #         keys.remove('/exprb3')
    #         Integrators = {key:IntegratorData(filelocation,key) for key in keys}
    #     for key in keys:
    #         get_optimal_data(Integrators[key], float(error), errortype, param)
        
    #     ax = axes[k]
    #     for key in keys:
    #         df = Integrators[key].optimaldata
    #         df.plot('Nx','m', label=key[1:], ax=ax)
    #     ax.set_xlabel('{{$N$}}')
    #     if k == 0:
    #         ax.set_title('Half precision')
    #         ax.set_ylabel('Jacobian-vector products per time step')
    #         ax.get_legend().remove()
    #     else:
    #         ax.legend()
    #         ax.set_title('Single precision')
    # fig.align_ylabels()
    # fig.subplots_adjust(top=0.93)
    
    # savefig(3, save, precision_type, savetext)
    
    
    '''
    1. Multi plot of 1.1 and 1.2
    '''
    
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey='row',
                         figsize=((width, width)))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(suptitle)
    
    for k, error in enumerate([2**-10,2**-24]):
        with pd.HDFStore(filelocation) as hdf:
            keys = hdf.keys()
            try:
                keys.remove('/exprb3')
            except ValueError:
                pass
            Integrators = {key:IntegratorData(filelocation,key) for key in keys}
        for key in keys:
            get_optimal_data(Integrators[key], float(error), errortype, param)
        
        rk2cost = Integrators['/rk2'].optimaldata.cost
        for key in keys:
            cost = Integrators[key].optimaldata.cost
            Integrators[key].optimaldata['costnormalized'] = cost/rk2cost
         
        
        ax = axes[0+k]
        for key in keys:
            df = Integrators[key].optimaldata
            df.plot('Nx','costnormalized', label=key[1:], ax=ax)
        #ax.set_ylim([5e-2,5e1])  
        ax.set_xlabel('{{$N$}}')
        ax.set_yscale('log')
        if k == 0:
            ax.set_title('Half precision')
            ax.set_ylabel('Normalized bytes\n read or written')
        else:
            ax.set_title('Single precision')
        ax.get_legend().remove()
            
        ax = axes[2+k]
        
        for key in keys:
            df = Integrators[key].optimaldata
            df.plot('Nx','tau', label=key[1:], ax=ax)
            
        ax.plot(df.Nx, CFL, label="$CFL$",linestyle=':')
        ax.set_xlabel('{{$N$}}')
        if k == 0:
            ax.set_ylabel('Optimal\n time step')
        ax.get_legend().remove()
        ax.set_yscale('log')
        
        ax = axes[4+k]
        for key in keys:
            df = Integrators[key].optimaldata
            df.plot('Nx','mv', label=key[1:], ax=ax, logy=True)
        ax.plot(df.Nx, theoreticalmv,linestyle='-.')
        ax.set_xlabel('{{$N$}}')
        if k == 0:
            ax.set_ylabel('Total\n Jacobian-vector products')
            ax.get_legend().remove()
        else:
            ax.legend(loc = 'upper left')
        
        ax.set_ylim([10,1200000])    
        
    fig.align_ylabels()
    fig.subplots_adjust(top=0.95)
    
    savefig('multi', save, savetext)
    
