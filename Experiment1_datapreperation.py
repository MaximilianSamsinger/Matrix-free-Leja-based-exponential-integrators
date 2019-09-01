import pandas as pd
import numpy as np

'''
Assumption:
    t_end = 0.1
'''
'''
Goal:
    Prepare data for plotting in Experiment1_plots.py
'''

class dataobject:
    def __init__(self,key):
        with pd.HDFStore('Experiment1.h5') as hdf:
            self.data = hdf[key]
        self.data['tau'] = 0.1 / self.data['substeps']
        self.data['gridsize'] = 1. / (self.data['Nx'] - 1)
        self.data['Pe'] = self.data.adv / self.data.dif
        self.data['pe'] = self.data.adv*self.data.gridsize / (2*self.data.dif)
        self.data['m'] = self.data['mv']/self.data['substeps']
        self.key = key
        self.name = key[1:]
        self.Nx = self.data['Nx'].unique()
        self.substeps = self.data['substeps'].unique()
        self.Pe = self.data['Pe'].unique()
        self.adv = self.data['adv'].unique()
        self.dif = self.data['dif'].unique()

def get_optimal_data(dataobj, maxerror, errortype, Peclettype, adv, dif,
                     subset='Nx', costs='mv',):
    assert(adv <= 1 or dif <= 1)
    data = dataobj.data
    data = data.loc[data[errortype] <= maxerror]
    
    if Peclettype == 'Pe':
        data = data.loc[(data['adv'] == adv) & (data['dif'] == dif)] 
    elif Peclettype == 'pe':
        # Study why 1e-13 is necessary and not something like 1e-15 for pe == 10.0
        data = data.loc[np.abs(data['pe'] - adv/dif) < 1e-13] 
    else:
        raise TypeError("Petype needs to be either 'pe' or 'Pe'")
        
    data = data.sort_values(by=costs)
    data = data.drop_duplicates(subset=subset, keep='first')
    dataobj.optimaldata = data.sort_values(by=subset)
    dataobj.optimaldata.reset_index(drop=True, inplace=True)