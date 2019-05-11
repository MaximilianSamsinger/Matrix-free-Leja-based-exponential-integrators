import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
        self.data['tau'] = 0.1 / self.data['Nt']
        self.data['Pe'] = self.data.adv / (self.data.dif * (self.data.Nx - 1))
        self.data['m'] = self.data['mv']/self.data['Nt']
        self.key = key
        self.name = key[1:]
        self.Nx = self.data['Nx'].unique()
        self.Nt = self.data['Nt'].unique()
        self.Pe = self.data['Pe'].unique()
        self.adv = self.data['adv'].unique()
        self.dif = self.data['dif'].unique()

def get_optimal_data(dataobj, tol, errortype, adv, dif,
                     subset='Nx', costs='mv',):
    data = dataobj.data
    data = data.loc[(data['adv'] == adv) 
              & (data['dif'] == dif)
              & (data[errortype]<tol)] 
    data = data.sort_values(by=costs)
    data = data.drop_duplicates(subset=subset, keep='first')
    dataobj.optimaldata = data.sort_values(by=subset)
    dataobj.optimaldata.reset_index(drop=True, inplace=True)