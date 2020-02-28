import pandas as pd
import matplotlib.pyplot as plt

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

def global_plot_parameters(SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,figsize):
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
            self.α = self.data['α'].unique()
            self.β = self.data['β'].unique()
            self.γ = self.data['γ'].unique()
            self.data['cost'] = (
                0*self.data.dFeval
                + self.data.Feval
                + self.data.mv
            )
        self.data['m'] = self.data['cost']/self.data['substeps']

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
    