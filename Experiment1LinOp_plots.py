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

Experiment 1:
    Here we fix the highest acceptable error (2-norm/inf-norm) for the solution
    of the linear advection-diffusion equantion and look at all datapoints
    for which an integrator satisfies:
        - The maximal integration error is small enough
        - The costs (matrix-vector multiplication) are minimal

    The integrators are
        - rk2... Runge-Kutta method of order 2
        - rk4... Runge-Kutta method of order 4
        - cn2... Crank-Nicolson method of order 2
        - exprb2... Exponential euler method of order 2, in this case EXACT

Note: We write expeuler, even though we only compute the matrix exponential
    function of the discretized matrix using expleja with single precision
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

save = False # Flag: If True, figures will be saved as pdf
save_path = 'figures' + os.sep + 'Experiment1' + os.sep

adv = 1.0 # Coefficient of advection matrix. Do not change
difs = [1e0,1e-1,1e-2]
dif = difs[0]

'''
1.5 Plot Matrix-vector multiplications (Nx) vs optimal time step size (tau)
'''


def savefig(number, save=False,
                add_to_name = ', Pe='+str(adv/dif)):
        if save:
            print('wow, it worked')
            plt.savefig(save_path + str(number) + ', ' + precision_type
                        + add_to_name + ".pdf"
                        , format='pdf', bbox_inches='tight', transparent=True)
            plt.close()



fig, ax = plt.subplots()
from copy import deepcopy

Nts = [250, 500, 750, 1000]
powerits = [2,3,4,6,8,10,25,50]

dobj = deepcopy(dataobjdict['/exprb2'])

for Nt in Nts:
    for dif in difs:
        fig, ax = plt.subplots()
        title = 'Péclet '
        for its in powerits:
            data = dobj.data
            data = data.loc[(data['misc']==its) & (data.substeps == Nt) &
                            (data.dif == dif) & (data.target_error == 2**-24)]
            data = data.drop_duplicates(subset='Nx', keep='first')
            data = data.sort_values(by='Nx')
            data.m = (data.mv-data.misc)/data.substeps + data.misc #Assumption: Matrix changes every substep
            try:
                data.plot('Nx','m',label=str(its) +' it', ax=ax)
            except TypeError:
                print('No numeric data to plot')
                print('Nt: %s, dif: %s, its: %s\n' %(Nt,dif,its))


        ax.legend()
        ax.set_xlabel('Matrix-vector multiplications')
        ax.set_ylabel('Optimal time step')
        ax.set_ylim([0,400])
        ax.set_ylim([0,120])

        savefig(5, save, '')

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
        get_optimal_data(dataobjdict[key], float(maxerror), errortype, dif)
        df = dataobjdict[key].optimaldata

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
'''
Experiment 1.4 (Bonus):

Compute the error of the expleja method for varying matrix dimensions NxN.
Normally this would result would be almost exact, but in this case we fix the
number of substeps and assume single precision.
'''
'''
fig, ax = plt.subplots(1, 1, sharex=True)
fig.suptitle(suptitle + '\n Single precision expleja')

data = dataobjdict['/exprb2'].data
data = data.loc[((data['dif'] == dif)
                & (data[errortype] <= 1)
                & (data['target_error'] == float(maxerror))]
data = data.sort_values(by='substeps')

for label, df in data.groupby('Nx'):
    if label%100 == 0:
        df.plot('substeps',errortype, ax=ax, label='N = ' + str(label))
ax.set_xlabel('Number of substeps')
if errortype == 'rel_error_inf':
    ax.set_ylabel('Relative error in maximum norm')
elif errortype == 'rel_error_2':
    ax.set_ylabel('Relative error in Euclidean norm')
ax.set_xscale('log')
ax.set_yscale('log')

plt.subplots_adjust(hspace=0)

savefig(4, save)
'''