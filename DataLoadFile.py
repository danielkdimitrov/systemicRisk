"""
Created on Thu Nov  5 17:35:43 2020

Load and plot the data

@author: daniel dimitrov (daniel.k.dimitrov@gmail.com)
Please quote paper : Quantifying Systemic Risk in the Presence of Unlisted Banks: Application to the Dutch Financial Sector
"""
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy import optimize

import statsmodels.formula.api as smf

from setParams import SetParams

class DataLoad(SetParams):
    def __init__(self):
        'Loading up the data'
        #'NN : too short of a time history - add later'
        self.universe = SetParams().universeFull # ['Aegon','ABN','NN'] # CDS : ['ABNsub','ING','Rabo','NIBC','Volksbank', 'AEGON'] 
        #  self.setConstants()
        'TODO : Put this with the Merton Class'
        
    def getCDS(self):
        cdsPrice = self.loadAllFiles('data\cds','w')
        cdsPrice = cdsPrice #in decimals 
        cdsLC = self.getLogChanges(cdsPrice)
        return cdsPrice, cdsLC
    
    def getDebt(self):
        return self.loadAllFiles('data\debt','a')
    
    def getVola(self):
        return self.loadAllFiles('data\\vola','w')   
        
    def loadAllFiles(self,subFolder,freq):
        '''
        folder : string of the child folder where the csv files are
        '''
        df = pd.DataFrame()
        for file_name in glob.glob(subFolder+'\*.csv'):
            print('Loading Data: ', file_name)
            series = pd.read_csv(file_name, header=0, parse_dates=["Date"], index_col=0, squeeze=True)
            if df.empty == False:
                df = pd.merge(df,pd.DataFrame(series) , on='Date', how='outer')
            else: 
                df = pd.DataFrame(series)
        if freq == 'w':
            'Keep Weekly data only : '
            #freq = freq # now, either 'w' for weekly or otherwise it's daily by default
            df = self.getWeekly(df)
        'TODO : if a (which is debt) fill in missing values to daily/weekly'
        'sort so that newest ? are on top : '
        df.sort_index(ascending=False, inplace=True)
        # else keep as it is
        return df
    
    def getWeekly(self,df):
        'Resample the daily Price data to weekly. Resmapling is done to keep Monday prices '
        offset = pd.offsets.timedelta(days=-6)
        df = df.resample('W',loffset=offset).first()
        df.sort_index(ascending=False, inplace=True)
        
        return df
            
    def getLogChanges(self, dfRaw):
        df = pd.DataFrame(index=dfRaw.index)
        
        'Get Log Returns'
        df = dfRaw.transform(lambda x: np.log(x)).diff(-1)
        'define the system. TODO : I need to be taking weights into account later on'
        df['Sys'] = df[self.universe].mean(numeric_only=True, axis=1)
                
        return df

class DataTransform(DataLoad):
    def __init__(self, getEquity=False):
        'Start to End data for the Backtests'
        self.endDate =  SetParams().firstDate #indxDates.max()
        self.startDate = SetParams().lastDate  #indxDates.min() 
    
        'Load : 1/ CDS price, and log-changes or Market Value'
        self.CDSprices, self.CDSreturns = DataLoad().getCDS()
        
        if getEquity == True:
            'Load : 2/ Market Value and Equity Returns'        
            self.eqMV = DataLoad().getEquityMV()
            self.eqReturns, self.eqPrices = DataLoad().getEquity()
            self.eqStd = self.eqReturns.sort_index(ascending=True).rolling(52).std()*np.sqrt(52) #250 50 weeks rolling, scale to a year
            self.eqStd.sort_index(ascending=True, inplace=True)            
         
        'Load : 3/ Debt, Recovery Rate, Deposits to Liabs Ratio'        
        debt = DataLoad().getDebt()
        self.debt = self.getToWeekly(debt)
        RR = pd.read_csv('data\\other\\RR.csv', header=0, parse_dates=["Date"], index_col=0, squeeze=True)
        self.RR = self.getToWeekly(RR)
        DL = pd.read_csv('data\\other\\DepositsToLiabs.csv', header=0, parse_dates=["Date"], index_col=0, squeeze=True)
        self.DL = self.getToWeekly(DL)
        
        'Get : Rolling StDev./Weekly'
        self.CDSstd = self.CDSreturns.sort_index(ascending=True).rolling(52).std()*np.sqrt(52) #250 50 weeks rolling, scale to a year
        self.CDSstd.sort_index(ascending=True, inplace=True)
        self.vola = DataLoad().getVola()/100
        
    def getToWeekly(self, df):
        '- Create empty row'
        df = df.append(pd.Series(name=self.endDate)) #
        df.sort_index(ascending=False, inplace=True)
        '- Get down to weekly and fill in missing'
        df = DataLoad().getWeekly(df)
        df = df.interpolate(method='quadratic')
        df = df.bfill() #
        return df
