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

Experiment1 Powerit:
    We study the matrix-free case and determine for which parameter
    (number of power iterations, substeps, etc) expleja still converges

    The integrators are
        - exprb2... Exponential euler method of order 2, in this case EXACT

Note: We write exprb2, even though we only compute the matrix exponential
    function of the discretized matrix using expleja with half/single precision
    and fixed step size.
'''

'''
Global plot parameters
'''
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#plt.style.use('seaborn')

width = 426.79135/72.27
fig_dim = lambda fraction: (width*fraction, width*fraction*0.75)

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['text.latex.unicode']=True
plt.rc('text', usetex=True)
plt.rcParams['figure.figsize'] = fig_dim(0.9)
plt.rc('font', family='serif')

'''
CONFIG
'''
maxerror = str(2**-24)
save = True # Flag: If True, figures will be saved as pdf
save_path = 'figures' + os.sep + 'Experiment1LinOp' + os.sep
adv = 1.0 # Coefficient of advection matrix. Do not change.
difs = [1e0, 1e-1, 1e-2] # Coefficient of diffusion matrix. Should be <= 1

'''
Load data
'''
filelocation = 'HDF5-Files' + os.sep + 'Experiment1LinOp.h5'
with pd.HDFStore(filelocation) as hdf:
    keys = hdf.keys()
    Integrators = {key:IntegratorData(filelocation,key) for key in keys}
    

'''
For saving and plotting
'''
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

if maxerror == str(2**-24):
    precision = '$2^{-24}$'
    precision_type = 'single'
elif maxerror == str(2**-10):
    precision = '$2^{-10}$'
    precision_type = 'half'

errortype = 'relerror'




from copy import deepcopy

Nts = [250, 500, 750, 1000]
powerits = [2,3,4,6,10,25]
safetyfactors = [0.75,0.9,1.,1.1,1.5]
tol = float(maxerror)

Integrator = deepcopy(Integrators['/exprb2'])

'''
1.1 Experiment 1
'''
mng = plt.get_current_fig_manager()
for dif, sf in product(difs, safetyfactors):
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()
    fig.suptitle('Cost per timestep for {{$\\mathrm{Pe}$}}' + f' = {1/dif} and sf = {sf}')
    for k, Nt in enumerate(Nts):
        ax = axes[k]
        ax.set_title(f'{{$s$}} = {{{Nt}}}')

        for its in powerits:
            ''' Prepare data '''
            data = Integrator.data
            data = data.loc[(data['powerits']==its) & (data.substeps == Nt) &
                            (data.dif == dif) & (data.safetyfactor == sf)
                            & (data.tol == tol)]
            data = data.drop_duplicates(subset='Nx', keep='first')
            data = data.sort_values(by='Nx')
            data.m = (data.mv-data.powerits)/data.substeps + data.powerits #Assumption: Matrix changes every substep
            ''' Here we try to plot '''
            try:
                data.plot('Nx','m',label=f'n={str(its)}', ax=ax,)
            except TypeError:
                print('No numeric data to plot')
                print('Nt: %s, dif: %s, its: %s\n' %(Nt,dif,its))

        if k==0:
            ax.legend(loc ='lower right')
        else:
            ax.get_legend().remove()
        if k in [0,1]:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('{{$N$}}')
        if k%2 == 0:
            ax.set_ylabel('{{$\\texttt{mv}/s$}}')
        else:
            ax.set_ylabel('')
        ax.set_xlim([50,400])
        ax.set_ylim([0,120])
    plt.subplots_adjust(hspace=0.35)
    savefig(1, save, f'Pe={1/dif}, sf={sf}')