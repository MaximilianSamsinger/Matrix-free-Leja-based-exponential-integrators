import pandas as pd

'''
Assumption:
    t_end = 0.1
'''
'''
Goal:
    Prepare data for plotting in ExperimentX_plots.py files
'''

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
            # 1/3 Derivative = 1 functionEvaluations = 3 mv
            self.data['cost'] = (
                0*self.data.dFeval
                + self.data.Feval
                + self.data.mv
            )
        self.data['m'] = self.data['cost']/self.data['substeps']

'''
F        30
dF(u)   325
dFu @ v  13.3
mv        8
'''

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
    
