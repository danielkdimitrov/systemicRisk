# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:41:39 2021

@author: daniel dimitrov (daniel.k.dimitrov@gmail.com)
Please quote paper : Quantifying Systemic Risk in the Presence of Unlisted Banks: Application to the Dutch Financial Sector
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats

from setParams import SetParams
from DataLoadFile import *
from SystemicRisk import *
from myplotstyle import * 
import seaborn as sns

'-------- Run one date only ---------------'
dataPresent = 'DataSet' in locals() or 'DataSet' in globals()

DataSet = DataTransform(getEquity = False)
    
'Plot data = True or False ?'
plotInd = True
'Set Universe and Time Window'
Params = SetParams()
universeFull = Params.universeFull
lastDate, firstDate = Params.lastDatePiT, Params.firstDatePiT 

mycolors = ['k', 'tab:blue', 'tab:grey', 'tab:orange', 'tab:brown', 'tab:green', 'tab:pink', 'tab:olive']   
myLineStyle=['-','--','-.',':','^-']   

'----- Get Liab Weights -----'
Debt = DataSet.debt.loc[lastDate:firstDate, Params.universeFull]
Debt['Sys'] = Debt.sum(axis=1)
wt = pd.DataFrame(index = Debt.index, columns = Debt.columns)
wt = Debt.div(Debt['Sys'], axis=0) 

'0. Get raw CDS rates'
dfCDSraw = DataSet.CDSprices.loc[lastDate:firstDate, Params.universeFull]

'0. Get CDS rate log changes'
dfCDSr = dfCDSraw.transform(lambda x: np.log(x)).diff(-1)

'0. Transform CDS data ' 
dfCDS = dfCDSraw/1e4
dfCDS.interpolate(method='quadratic', inplace=True)
dfRR = DataSet.RR.loc[lastDate:firstDate, Params.universeFull]

'1. Get PD, DD data'        
pdd = PD(dfCDS,dfRR) 

'Get Distance-to-Default DD log changes'
dfU = pdd.dfDD.transform(lambda x: np.log(x.astype('float64'))).diff(-1)
dfU = dfU[:-1] # drop the first NoN obs

'2. Get Factor Model'
fm = FactorModel(dfU)
alpha = pd.DataFrame(fm.ldngs, index=universeFull, columns=['Loadings'])

if plotInd == True:
    'Factor Exposure Spider Chart'
    lollipopChart(alpha, 'firebrick')
        
    'Asset Corrs'
    corrs = alpha@alpha.T
    corrs_df = pd.DataFrame(corrs, columns = universeFull, index=universeFull)
    fig, axs = plt.subplots(1, 1,figsize=(7,6))
    mask = np.triu(np.ones_like(corrs_df, dtype=np.bool))     
    sns.heatmap(corrs_df, mask=mask, annot=True, fmt=".2f", cmap='Blues',ax = axs)

############################################
'3. get simulation of the losses. '
debtEval = DataSet.debt[Params.universeFull].loc[lastDate].squeeze(axis=0) #.loc[evalDate] #.copy()
debtEval['Sys'] = debtEval.sum()
        
DDEval = pdd.dfDD.loc[lastDate].squeeze(axis=0)
ERREval = dfRR.loc[lastDate].squeeze(axis=0)
sigmaC = DataSet.vola.loc[lastDate,'vstoxx'].values
lsim = LossesSim(sigmaC, fm.ldngs, debtEval, DDEval, ERREval, 'PCD')

'4. get PDs (default probab.s)'

pds = DefaultP(lsim.IndctrD, True)
jpd = pds.jpd.astype(float)*100    
cpd = pds.cpd.astype(float)*100    

if plotInd == True:
    
    'Conditional Probab. Default Matrix'
    ax = myplot_frame((12,4))
    mask = np.invert(np.ones_like(cpd, dtype=np.bool))    
    np.fill_diagonal(mask,True)    
    sns.heatmap(cpd,fmt=".2f", annot=True, cmap='Oranges', mask= mask, alpha=0.8, ax = ax)
    ax.set_xlabel(r'Conditional on $j$',fontsize=12)
    ax.set_ylabel(r'$i$ term',fontsize=12)    

    'Joint Probab. Default Matrix'
    axs= myplot_frame((12,4))
    mask = np.invert(np.tril(np.ones_like(jpd, dtype=np.bool)))     
    sns.heatmap(jpd, mask=mask, annot=True, fmt=".2f", cmap='Oranges',alpha=0.8, vmin=0, vmax=1, ax = axs)

'5. Get VaR, ES, MES estimates '
covarr = CoVaR(lsim.Lsim)
wts = debtEval/debtEval['Sys'] 

print('ES:', covarr.ES.T)
print('MES:', covarr.MES.T)
        
'// Main Table with Results'

tableMain = pd.DataFrame(index=covarr.MES.columns, columns=['EL','ELr', 'w', 'wr', 'ES99','ES99r','MES99','MES99r','PCtoES99','PCtoES99r'])
tableMain['EL'] = covarr.Exptn
tableMain['w'] = wts   
tableMain['MES99']  = covarr.MES.loc[.99]
tableMain['ES99'] = covarr.ES.loc[.99]
tableMain['PCtoES99']  = 100*covarr.MES.loc[.99]*wts/covarr.ES.loc[.99]['Sys']

tableMain['ELr'] = tableMain['EL'].rank(ascending=False)
tableMain['wr'] = tableMain['w'].rank(ascending=False)
tableMain['MES99r']  = tableMain['MES99'].rank(ascending=False)
tableMain['ES99r'] = tableMain['ES99'].rank(ascending=False)
tableMain['PCtoES99r']  = tableMain['PCtoES99'].rank(ascending=False)

'Print table in Excel'
tableMain.to_excel('tableMain.xlsx')

'// Network : ES Matrix'
print('ES Matrix .99: \n', np.round(covarr.ExS99, 2))
covarr.ExS99.to_excel('NetworkESMatrix.xlsx')

'Heatmap Exposure Shortfall' 
ax = myplot_frame((12,4))
sns.heatmap(covarr.ExS99, annot=True, fmt=".3g", cmap='OrRd', alpha=0.8, ax=ax)
ax.set_xlabel(r'Conditional on $j$',fontsize=12)
ax.set_ylabel(r'$i$ term',fontsize=12)    
       
    
    