import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from itertools import chain, product
from Experiment1_datapreperation import dataobject, get_optimal_data

'''
DISCRETIZED ONE DIMENSIONAL LINEAR ADVECTION-DIFFUSION EQUATION

Experiment 1LinOP:
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
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['lines.linewidth'] = 3

'''
Load data
'''
filename = 'Experiment1LinOp.h5'
with pd.HDFStore(filename) as hdf:
    keys = hdf.keys()
    dataobjdict = {key:dataobject(filename,key) for key in keys}

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

errortype = 'rel_error_2norm'

save = True # Flag: If True, figures will be saved as pdf
save_path = 'figures' + os.sep + 'Experiment1LinOp' + os.sep

adv = 1.0 # Coefficient of advection matrix. Do not change
difs = [1e0,1e-1,1e-2]
dif = difs[0]


fig, ax = plt.subplots()
from copy import deepcopy

Nts = [250, 500, 750, 1000]
powerits = [2,3,4,6,10,25]
safetyfactors = [0.75,0.9,1.,1.1,1.5]
target_error = 2**-24

dobj = deepcopy(dataobjdict['/exprb2'])

def savefig(number, save=False,
            add_to_name = ''):
    if save:
        plt.savefig(save_path + f'{number}, ' + precision_type
                    + add_to_name + ".pdf"
                    , format='pdf', bbox_inches='tight', transparent=True)
        print('File saved')
        plt.close()

'''
1.1 Experiment 1
'''
mng = plt.get_current_fig_manager()
for dif, sf in product(difs, safetyfactors):
    fig, axes = plt.subplots(2,2, figsize=(12,9))
    axes = axes.flatten()
    fig.suptitle(f'Péclet: {1/dif}, sf: {sf}')
    for k, Nt in enumerate(Nts):
        ax = axes[k]
        ax.set_title(f's: {Nt}')

        for its in powerits:
            ''' Prepare data '''
            data = dobj.data
            data = data.loc[(data['misc']==its) & (data.substeps == Nt) &
                            (data.dif == dif) & (data.safetyfactor == sf)
                            & (data.target_error == target_error)]
            data = data.drop_duplicates(subset='Nx', keep='first')
            data = data.sort_values(by='Nx')
            data.m = (data.mv-data.misc)/data.substeps + data.misc #Assumption: Matrix changes every substep
            ''' Here we try to plot '''
            try:
                data.plot('Nx','m',label=str(its) +' it', ax=ax,)
            except TypeError:
                print('No numeric data to plot')
                print('Nt: %s, dif: %s, its: %s\n' %(Nt,dif,its))

        ax.legend(loc ='lower right')
        ax.set_xlabel('N')
        ax.set_ylabel('mv/s')
        ax.set_xlim([50,400])
        ax.set_ylim([0,120])
    plt.subplots_adjust(hspace=0.35)
    savefig(1, save, f', Pe={1/dif}, sf={sf}')



