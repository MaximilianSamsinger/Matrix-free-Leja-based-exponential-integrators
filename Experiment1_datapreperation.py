import pandas as pd

'''
Assumption:
    t_end = 0.1
'''
'''
Goal:
    Prepare data for plotting in Experiment1_plots.py
'''

class dataobject:
    def __init__(self,filename,key):
        with pd.HDFStore(filename) as hdf:
            self.data = hdf[key]
        self.data['tau'] = 0.1 / self.data['substeps']
        self.data['gridsize'] = 1. / (self.data['Nx'] - 1)
        self.data['Pe'] = 1.0 / self.data.dif
        self.data['m'] = self.data['mv']/self.data['substeps']
        self.key = key
        self.name = key[1:]
        self.Nx = self.data['Nx'].unique()
        self.substeps = self.data['substeps'].unique()
        self.Pe = self.data['Pe'].unique()
        self.dif = self.data['dif'].unique()

def get_optimal_data(dataobj, maxerror, errortype, dif,
                     subset='Nx', costs='mv',):
    assert(dif <= 1)
    data = dataobj.data
    data = data.loc[data[errortype] <= maxerror]
    
    data = data.loc[data['dif'] == dif] 
        
    data = data.sort_values(by=costs)
    data = data.drop_duplicates(subset=subset, keep='first')
    dataobj.optimaldata = data.sort_values(by=subset)
    dataobj.optimaldata.reset_index(drop=True, inplace=True)