# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:38:28 2021

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

from setParams import SetParams
from CoVaRDataLoad import *
from FactorModel import *
from myplotstyle import saveFig, myplot_frame


'--------------- Run on a Rolling Window ------------'

'Load Data'
dataPresent = 'DataSet' in locals() or 'DataSet' in globals()
if dataPresent == False: 
    DataSet = DataTransform(getEquity = False)

mycolors = ['k', 'tab:blue', 'tab:grey', 'tab:orange', 'tab:brown', 'tab:green', 'tab:pink', 'tab:olive']   
myLineStyle=['-','--','-.',':','^-']   
path = "C:\\Users\\danie\\Dropbox\\Amsterdam\\CoVaR_Project\\images\\"
Params = SetParams()

rwcovar = RollingWindowCoVaR(DataSet,'PCD', True)

axs = myplot_frame((12,6))
rwcovar.ES99['Sys'].plot(color='black',ax=axs,label='ES99')
rwcovar.VaR99['Sys'].plot(color='black',linestyle=':',ax=axs, label='VaR99')
axs3.set_xlabel('')
axs.legend(fontsize='xx-large')
saveFig(path,'VaRESsys')

'Plot VaR, CoVaR, '
'''
rwcovar.VaR.plot(alpha=0.6, figsize=(9, 12),subplots=True) #sharey=True, , title='VaR 99%'
saveFig(path,'VaRs')

rwcovar.VaR50.plot(subplots=True, title='VaR 50%') #sharey=True,

rwcovar.CoVaR.plot(alpha=0.6, figsize=(9, 12), subplots=True) #, title='CoVaR 99%'
saveFig(path,'CoVaRs')

rwcovar.CoVaR50.plot(subplots=True, title='CoVaR 50%')

rwcovar.Weight.plot(subplots=True, title='Systemic Weight')

rwcovar.DeltaCoVaR.plot(subplots=True, title='DeltaCoVaR (%)') #, sharey=True 

'''

'''
'--------- For the comparison sigma_c chart ---------'
rwcovar_v02 = RollingWindowCoVaR(DataSet,'Sys', True) #fixing sigma_c = .15
rwcovar_v03 = RollingWindowCoVaR(DataSet,'Sys', True) #fixing sigma_c = .05

MESvstoxx = rwcovar.MES99["Sys"]  
MESvstoxx_v02 = rwcovar_v02.MES99["Sys"]
MESvstoxx_v03 = rwcovar_v03.MES99["Sys"]
'''
vstoxx = DataSet.vola.loc[:'2012-01-01','vstoxx']*100

'''MES, Stacked area  ''' 

axs = myplot_frame((12,6))
MESvstoxx.plot(ax=axs,label='$\sigma_c=VSTOXX$', linestyle=myLineStyle[1])
MESvstoxx_v02.plot(ax=axs,label=r'$\sigma_c=.15$', linestyle=myLineStyle[2])
MESvstoxx_v03.plot(ax=axs,label=r'$\sigma_c=.05$', linestyle=myLineStyle[2])
axs.set_ylabel('Systemic Loss, %', size=16)

axs.legend()
saveFig(path,'MESComparedSigmaC')

'---------------'
#vstoxx.plot(alpha=0.6, ax=axs,label='VSTOXX', linestyle=myLineStyle[0])


fig, axs = plt.subplots(3, 1,figsize=(12,6))

axs = myplot_frame((8,6))
rwcovar.MES["Sys"].plot(ax=axs[0],label='ES System')
DataSet.vola.loc[:'2012-01-01','vstoxx'].plot(alpha=0.6, ax=axs[0],label='VSTOXX')
rwcovar.MES[Params.universeFull].loc[:'2012-01-01'].plot.area(ax=axs[1])
rwcovar.pN.plot(ax=axs[2])
axs[0].legend()


###
'''MES, per institution  ''' 

'''
axs = myplot_frame((8,6))
rwcovar.MES[Params.universeFull].loc[:'2012-01-01'].plot.area(alpha=0.6, ax=axs)
saveFig(path,'MES')

axs = myplot_frame((8,6))
rwcovar.MES99[Params.universeFull].loc[:'2012-01-01'].plot.area(alpha=0.6, ax=axs)
saveFig(path,'MES99')

axs = myplot_frame((8,6))
rwcovar.MES[Params.universeFull].loc[:'2012-01-01'].plot(subplots=True, sharey=True, sharex=True, ax=axs)
#rwcovar.MES[Params.universeFull].loc[:'2012-01-01'].plot(subplots=True, sharey=True, sharex=True, ax=axs)
saveFig(path,'MES')
'''

'Percentage contributions 99'
axs = myplot_frame((12,6))
df = rwcovar.MES99[Params.universeFull].loc[:'2012-01-01']
df = df.divide(df.sum(axis=1), axis=0)
df.sort_values(by=df.index[0],axis=1).plot.area(cmap = 'Oranges', ax= axs) #
axs.set_xlabel('')
saveFig(path,'PCMES99')



### 
'''
fig, axs1 = plt.subplots(1, 1,figsize=(12,6))
rwcovar.ES.loc[:'2012-01-01'].plot(subplots=True)
rwcovar.ES99.loc[:'2012-01-01'].plot(subplots=True)
'''
#######################

fig, axsDCVr = plt.subplots(6, 1,figsize=(12,6))
rwcovar.DeltaCoVaR[Params.universeFull].plot() #,   title='DeltaCoVaR (%)'

####################### MES

'MES, ES per institution'

fig, axs = plt.subplots(7, 1,figsize=(12,6))
for jN,name in enumerate(Params.universeFull):
    rwcovar.ES99[name].plot(color='black', ax = axs[jN], label=r'$ES99$')
    rwcovar.MES99[name].plot(color='grey', linestyle='-.',ax = axs[jN], label=r'$MES99$')   
        
    axs[jN].set_xlabel('')
    #axs[jN].set_ylim([0,.12])
    axs[jN].set_title(name, y=1.0, pad=-14)
    if jN < 6:        
        axs[jN].set_xticks([])

    axs[jN].set_ylim([0,85])
lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'lower center', frameon=False, ncol=3)
saveFig(path,'ESMES') #VaRCoVaR


'PDs over time'
#pdd = PD(dfCDS,dfRR) 

fig, axs = plt.subplots(7, 1,figsize=(12,6))

for jN,name in enumerate(Params.universeFull):
    axs2 = axs[jN].twinx()
    #temp = pdd.dfPD[name]*100
    pdd.dfPD.multiply(100).loc[:'2012-01-01',name].plot(color='black',ax = axs[jN], label=r'$PD$')   
    rwcovar.SII.loc[:'2012-01-01',name].plot(color='grey', linestyle='-.', ax = axs2, label=r'$SII$')
        
    axs[jN].set_xlabel('')
    #axs[jN].set_ylim([0,.12])
    axs[jN].set_title(name, y=1.0, pad=-14)
    if jN < 6:        
        axs[jN].set_xticks([])

    axs2.set_ylim([0,5])
lines, labels = axs[0].get_legend_handles_labels()
lines2, labels2 = axs2.get_legend_handles_labels()

fig.legend(lines+lines2, labels+labels2, loc = 'lower center', frameon=False, ncol=3)
saveFig(path,'SII') #VaRCoVaR


####################### MES, DeltaCoVaR

fig, axs = plt.subplots(1, 1,figsize=(12,6))
rwcovar.VaR99['Sys'].plot(color='black', ax = axs, label='VaR')
rwcovar.ES99['Sys'].plot(color='black',linestyle='--', ax = axs, label='ES')

fig, axs2 = plt.subplots(1, 1,figsize=(12,6))
rwcovar.MES99[Params.universeFull].loc[:'2012-01-01'].mean(axis=1).plot(color='k', ax=axs2)
axs2.set_ylabel('Av. MES (solid line)', size=17)

axs_twin = axs2.twinx()
rwcovar.CoVaR99[Params.universeFull].loc[:'2012-01-01'].mean(axis=1).plot(linestyle='--', ax=axs_twin)
axs_twin.set_ylabel('Av. CoVaR (dashed)', size=17)

axs2.set_xlabel('')

saveFig(path,'SysVaRCoVaR99')


'Plot probabs'

axs3 = myplot_frame((12,6))
rwcovar.pN.plot(ax=axs3, color=mycolors, style=['-','--','-.',':','^-'], legend='off') #sharey=True, , title='VaR 99%' ,
axs3.set_xlabel('')
lines, labels = fig.axes[0].get_legend_handles_labels()
axs3.legend([r'$P(N_d \geq 1)$',r'$P(N_d \geq 2)$',r'$P(N_d \geq 3)$',r'$P(N_d \geq 4)$'],fontsize='xx-large')
axs3.set_ylabel('Joint PDs',fontsize='xx-large')
#fig.legend(lines, labels, loc = 'lower center', frameon=False, ncol=4)
saveFig(path,'pN')

### 
axs3 = myplot_frame((12,6))
rwcovar.pNafter1['p2'].plot(ax=axs3, color=mycolors, style=['-','--','-.',':','^-'], legend='off') #sharey=True, , title='VaR 99%' ,
axs3.set_xlabel('')
lines, labels = fig.axes[0].get_legend_handles_labels()
axs3.legend([r'$P(N_d \geq 2|N_d \geq 1)$',r'$P(N_d=3|N_d \geq 1)$',r'$P(N_d=4|N_d \geq 1)$'],fontsize='xx-large')
axs3.set_ylabel('Conditional PDs',fontsize='xx-large')
#fig.legend(lines, labels, loc = 'lower center', frameon=False, ncol=4)
saveFig(path,'pNafter1')

###
'''
axs3 = myplot_frame((8,6))
rwcovar.pNafter2['p3'].plot(ax=axs3, color=mycolors, style=['-','--','-.',':','^-'], legend='off') #sharey=True, , title='VaR 99%' ,
axs3.set_xlabel('')
lines, labels = fig.axes[0].get_legend_handles_labels()
axs3.legend([r'$P(N_d \geq 3|N_d \geq 2)$',r'$P(N_d=3|N_d \geq 2)$',r'$P(N_d=4|N_d \geq 2)$'])
axs3.set_ylabel('Conditional Probability of defaults')
#fig.legend(lines, labels, loc = 'lower center', frameon=False, ncol=4)
saveFig(path,'pNafter2')
'''

'''
print('CoVaR :', rwcovar.CoVaR.head().T)

print('VaR :', rwcovar.VaR.head().T)

'Plot Debt'
Debt = rwcovar.Debt
SysDebt = Debt.sum(axis=1)
rwcovar.Debt.plot.area(title='Debt',stacked=True)
'''
'--- Plot Loadings ---'
axsLd = myplot_frame((8,6))
rwcovar.LoadingsF1.plot(ax=axsLd, color=mycolors, style=['-','--','-.',':','--.'], legend='off') #sharey=True, , title='VaR 99%' , 
axsLd.set_xlabel('')
saveFig(path,'DataLoadings')

#axs3.linestyle(':')

#rwcovar.LoadingsF1.plot(title='Factor 1 Loadings')
#rwcovar.LoadingsF2.plot(title='Factor 2 Loadings')

#rwcovar.dfLoadingsF2.plot()
