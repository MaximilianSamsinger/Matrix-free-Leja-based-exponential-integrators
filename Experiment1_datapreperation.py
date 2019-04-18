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
        self.key = key
        self.name = key[1:]
        self.Nx = self.data['Nx'].unique()
        self.Nt = self.data['Nt'].unique()
        self.Pe = self.data['Pe'].unique()
        self.adv_coeff = self.data['adv_coeff'].unique()
        self.dif_coeff = self.data['adv_coeff'].unique()

def get_optimal_data(dataobj, tol, adv, dif, subset=['Nx'], costs='mv',):
    data = dataobj.data.sort_values(by=costs)
    data = data.loc[(data['adv_coeff'] == adv) 
                  & (data['dif_coeff'] == dif)
                  & (data['error']<tol)] 
    dataobj.optimaldata = data.drop_duplicates(subset=subset, keep='first')
    dataobj.optimaldata.reset_index(drop=True, inplace=True)

'''
def plot_under_condition(data, x, y, x_cond, y_cond, logplot,
                         label, title, figure_num):

        plt.figure(figure_num)
        X, Y = data[x], data[y]
        X = X[(x_cond[0] <= X) & (X <= x_cond[1])]
        Y = Y[(x_cond[0] <= X) & (X <= x_cond[1])]
        X = X[(y_cond[0] <= Y) & (Y <= y_cond[1])]
        Y = Y[(y_cond[0] <= Y) & (Y <= y_cond[1])]
        plt.plot(X, Y, label=label)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        plt.legend()
        
        if logplot[0]:
            plt.xscale('log')
        if logplot[1]:
            plt.yscale('log')

def x_vs_y(dataobj, x, y, by='Nx', data='data',
           logplot = [False, False],
           x_cond = [-float('inf'),float('inf')], 
           y_cond = [-float('inf'),float('inf')],
           figure_num = -1):
    
    data = getattr(dataobj, data)
       
    if by is None:
        plot_under_condition(data, x, y, x_cond, y_cond, logplot,
                                 label = dataobj.name, 
                                 title = 'tol 1e-8',
                                 figure_num = figure_num)
    else:
        column = getattr(dataobj, by)
        for entry in column:
            data_cond = data[data[by] == entry]
            plot_under_condition(data_cond, x, y, x_cond, y_cond, logplot,
                                 label = by + ': '+str(entry),
                                 title = dataobj.name,
                                 figure_num = figure_num)
'''