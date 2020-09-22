import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

'''
Assumption:
    t_end = 0.1
'''
'''
Goal:
    Prepare data for plotting in ExperimentX_plots.py files
'''

def savefigure(path, number, save=False, *add_to_filename):
    filename = f'{number}'
    for arg in add_to_filename:
        filename += f', {arg}'
    filename += '.pdf'
    print(filename)
    if save:
        plt.savefig(path + filename, 
                    format='pdf', bbox_inches='tight', transparent=True)
        print('File saved')
        plt.close()

def global_plot_parameters(SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,figsize,
                           LEGEND_SIZE=None):
    LEGEND_SIZE = SMALL_SIZE if LEGEND_SIZE==None else LEGEND_SIZE
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    plt.rcParams['lines.linewidth'] = 1.5
    try:
        plt.rcParams['text.latex.unicode']=True
    except KeyError:
        pass
    plt.rc('text', usetex=True)
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        default_colors[k] for k in [2,1,3,9,0,5]
    ])
    plt.rcParams['figure.figsize'] = figsize
    plt.rc('font', family='serif')

class IntegratorData:
    def __init__(self,filelocation,key):
        with pd.HDFStore(filelocation) as hdf:
            self.data = hdf[key]
        self.data['tau'] = 0.1 / self.data['substeps']
        self.data['gridsize'] = 1. / (self.data['Nx'] - 1)
        self.key = key
        self.name = key[1:]
        self.Nx = self.data['Nx'].unique()
        self.substeps = self.data['substeps'].unique()
        try:
            ''' Experiment 1 '''
            self.Experiment = 1
            self.dif = self.data['dif'].unique()
            #self.data['Pe'] = 1.0 / self.data.dif
            #self.Pe = self.data['Pe'].unique()
            self.data['cost'] = self.data['mv']
        except KeyError:
            ''' Experiment 2 '''
            self.Experiment = 2
            filename = os.path.split(filelocation)[-1]
            assert filename in ['Experiment2.h5', 'Experiment_2D.h5']
            d = 1 if filename=='Experiment2.h5' else 2
            self.α = self.data['α'].unique()
            self.β = self.data['β'].unique()
            self.γ = self.data['γ'].unique()
            
            filename = os.path.split(filelocation)[-1]

            
            self.data['cost'] = (
                0*self.data.dFeval # Number of times dF is initialized
                + 16*self.data.Feval # Number of times F is evaluated
                + 24*self.data.mv # Number of times dF(u)v is evaluated
            ) * (self.data.Nx**(d/2) * 2**(-10))**2 # Cost in megabyte
        self.data['m'] = self.data['mv']/self.data['substeps']

def get_optimal_data(integrator, maxerror, errortype, param,
                     subset='Nx'):
    
    data = integrator.data
    data = data.loc[data[errortype] <= maxerror]

    if integrator.Experiment == 1:
        ''' Experiment 1 '''
        data = data.loc[data['dif'] == param]
        cost = 'mv'
    elif integrator.Experiment == 2:
        ''' Experiment 2 '''
        data = data.loc[
                (data['α'] == param[0]) &
                (data['β'] == param[1]) &
                (data['γ'] == param[2])]
        cost = 'cost'
    else:
        return KeyError('Neither Experiment 1 nor Experiment 2')

    data = data.sort_values(by=cost)
    data = data.drop_duplicates(subset=subset, keep='first')
    integrator.optimaldata = data.sort_values(by=subset)
    integrator.optimaldata.reset_index(drop=True, inplace=True)
    
