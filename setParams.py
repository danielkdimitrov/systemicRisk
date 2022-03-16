# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:48:04 2021

@author: daniel dimitrov (daniel.k.dimitrov@gmail.com)
Please quote paper : Quantifying Systemic Risk in the Presence of Unlisted Banks: Application to the Dutch Financial Sector

"""

import pandas as pd
import numpy as np
import datetime

class SetParams:
        def __init__(self,evalDate='2009-12-28'):
            'Backtests Start / End Date :'
            self.lastDate = pd.to_datetime('2021-09-13')
            self.firstDate = pd.to_datetime('2010-01-01')            
            'Point in time estimates' 
            self.lastDatePiT = '2021-09-13',
            self.firstDatePiT = '2019-09-09'
            
            self.tw = 104#250
            self.dt = datetime.timedelta(weeks=self.tw)
            'Modelling parameters'
            self.q = .95
            self.nF = 1  # number of factors
            'Simulation Parameters'
            self.nSims = 5*10**5 
            self.df = 5
            self.universeFull = ['ABN', 'INGB','RABO','NIBC','VB','AEGO', 'NN'] 
                        
            'Merton :'
            self.r = 0.#0.005 
            self.T = 1. 
            'Data frequency'
            self.dt = 1/52
            
            'sytem'
            self.path = '/images' #the path for saving figures

