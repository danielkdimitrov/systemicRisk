# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:29:32 2021

@author: daniel dimitrov (daniel.k.dimitrov@gmail.com)
Please quote paper : Quantifying Systemic Risk in the Presence of Unlisted Banks: Application to the Dutch Financial Sector
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from setParams import SetParams
from pandas.plotting import scatter_matrix

from scipy.stats import norm, beta, triang
#from scipy.optimize import root, minimize, Bounds
from statsmodels.stats.correlation_tools import cov_nearest
#from sklearn.covariance import LedoitWolf

class PD:
    def __init__(self, dfCDSp, dfRR):
        '''
        Get Implied Probab of Default, and Distance to Default from CDS prices
        INPUT : 
            dfCDSp : dataframe - CDS prices : first divide by 1e4
        OUTPUT : 
            dfPD : dataframe - Implied (from CDS rates) Probability of Default
            dfDD : dataframe - Implied Distance to Default 
        '''
        
        'initialize'
        self.dfPD = pd.DataFrame(index=dfCDSp.index, columns = dfCDSp.columns)
        self.dfDD = pd.DataFrame(index=dfCDSp.index, columns = dfCDSp.columns)

        for jN, Name in enumerate(dfCDSp.columns):
            for indexDate, CDS in dfCDSp[Name].iteritems():
                RR = dfRR.loc[indexDate,Name]
                self.dfPD.loc[indexDate,Name], self.dfDD.loc[indexDate,Name] = self.getPD(CDS, 1 - RR)
            
    def getPD(self, CDS, LGD):
        '''
        r : risk-free rate 
        T : maturity of the CDS
        dt: descrete time period  
        '''
        r, dt, T  = .005, 1, 5 #LGD .8
        
        a = (1/r)*(1-np.exp(-T*dt*r) )
        b =  ((1/r)**2) *(1 - (T*dt*r + 1)*np.exp(-T*r*dt))
        'Probability of default:'
        PD = (a*CDS) / (a*LGD + b*CDS)
        'Distance to Default'
        DD = - norm.ppf(PD)
        return PD, DD  

class FactorModel:
    def __init__(self, dfDD):
        '''
        Algorithm to get the factor loadings of the model, based on Andersen 2003.
        INPUT : 
            df : Input from the PD class. It's the standard normal inverse of the probability of default,
            representing the unstandardized asset reutrns. 
        '''
        'standardize, so that the cov matrix later on is the corr matrix'
        dfStd = (dfDD -dfDD.mean())/dfDD.std()
        'get the loadings'
        self.ldngs, self.Cov = self.getLoadings(dfStd)
        'get the factors'
        self.fctrs = dfStd.dot(self.ldngs)

    def getLoadings(self, dfStd):
        '''
        Implementation of the Andersen 2003 Algorithm
        
        INPUT : 
            dfStd - standardized dataFrame containing the variables time series
        
        OUTPUT : 
            c - matrix of factor loadings, sum of squares less than 1
        '''
        def nearPSD(A):
            C = (A + A.T)/2
            eigval, eigvec = np.linalg.eig(C)
            eigval[eigval < 0] = 0
        
            return eigvec.dot(np.diag(eigval)).dot(eigvec.T)
           
        Cov = np.cov(dfStd.T)
        n = dfStd.columns.size # number of variables 
        F = np.diag(np.ones(n)*0.001)
        nF = SetParams().nF  # number of factors 
        
        eps, iters = 100, 0
        Niters = 150
        while (eps > 1e-6 and iters < Niters):
            iters += 1 
            SigmaF = Cov - F
            #SigmaF1 = cov_nearest(SigmaF) # to ensure cov matrix is positive definite
            eigenval, eigenvect = np.linalg.eig(SigmaF)
            if np.any(np.iscomplex(eigenval)) == True:
                print('complex egienvalues!')
                print(eigenval)
                break
            if np.any(eigenval[:nF] < 0) == True:
                print('Make SigmaF positive semi-definite')
                SigmaF = nearPSD(SigmaF)
                eigenval, eigenvect = np.linalg.eig(SigmaF)

            LambdaM =np.diag(eigenval[:nF])
            sqLambdaM = np.linalg.cholesky(LambdaM)
            
            E = eigenvect[:,:nF]
            c = np.matmul(E,sqLambdaM)
        
            cc = np.matmul(c, c.T)
            Fnew = np.diag(1- np.diag(cc))
            eps = np.linalg.norm(F-Fnew)
            F = Fnew.copy()
        if iters > Niters: 
            print('Factor model did not converged')
            print('eps:', eps)
        if c[0,0] < 0:
            c = -c
        #print('loadings', c)
        return c, Cov
    
class LossesSim:
    def __init__(self, sigmaC, ldngs, Debt, dfDD, dfERR, lossType, distrib = 'Normal', printt=False):
        '''
        INPUT : 
            dfDD - SERIES - distance to default
            fLoadings - NUMPY ARRAY - factor loadings 
            DD - SERIES - distance to default
        OUTPUT : 
            A matrix of simulated losses
        '''
        
        np.random.seed(1)
        
        Lsim = pd.DataFrame(columns = dfDD.index)
        self.RRsim = pd.DataFrame(columns = dfDD.index) #index=range(nSims),
        self.dWsim = pd.DataFrame(columns = dfDD.index) 
        self.IndctrD = pd.DataFrame(0, index=np.arange(SetParams().nSims), columns = dfDD.index) 
        
        self.ELGD = pd.DataFrame(index=[0], columns = dfDD.index)
        nA, nF = ldngs.shape # number of firms and number of factors
        'Get Factor Simulations'
        nSims = SetParams().nSims
        if distrib == 'Normal':
            M = np.random.normal(loc=0.0, scale=1.0, size=(nF, nSims))  #Factor
            dZ = np.random.normal(loc=0.0, scale=1.0, size=(nA, nSims))  # Idiosynch 
            dZc = np.random.normal(loc=0.0, scale=1.0, size=(nA, nSims))
        else:
            M = np.random.standard_t(SetParams().df, size = (nF, nSims))
            dZ = np.random.standard_t(SetParams().df, size=(nA, nSims))
            dZc = np.random.standard_t(SetParams().df, size=(nA, nSims))
        
        for jN, Name in enumerate(dfDD.index):
            'Simulate Factor Loadings'
                        
            A = ldngs[jN,:] # factor loadings for firm jN
            
            dW = M.T@A + np.sqrt(1- A.T@A)*dZ[jN,:]
            'Simulate RR : use only the first factor'
            C = M.T@A + np.sqrt(1- A.T@A)*dZc[jN,:] # this is dWc in the paper
            sigmaC = .15
            RR = np.minimum(1, np.exp(sigmaC*C) ) #Collateral #SetParams().
            mu_c = dfERR[Name]/np.mean(RR)
            RR = mu_c*RR

            if printt == True : print('Simulating:', Name)
            self.IndctrD[Name][dW <= - dfDD.loc[Name]] = 1
            
            Lsim[Name] = self.IndctrD[Name]*Debt.loc[Name]*(1-RR)
            self.ELGD.loc[0,Name] = Lsim[Name][self.IndctrD[Name]>0].mean()/Debt[Name] # expected loss given default
            self.RRsim[Name] = RR
            self.dWsim[Name] = dW
        
        'originally Lsim (losses) are in euro'
        Lsim['Sys'] = Lsim.sum(axis=1) #get systemic losses in each scenario
        
        if lossType == 'Sys':
            'Losses as percent of systemic debt (sum of all firms debt)'
            Lsim = 100*(Lsim/Debt.loc['Sys'])
            
        if lossType == 'PCD':
            'Losses as percent of debt'
            Lsim = 100*Lsim/Debt        
        
        self.IndctrD['Sys'] = self.IndctrD.sum(axis=1) #number of defaults        
        self.M = pd.DataFrame(M.T)
        self.Lsim = Lsim        
        self.Debt = Debt

class CoVaR:
    def __init__(self,dfLsim):
        '''
        INPUT : 
            dfLsim - dataframe - simulated Losses
            q - the tail probability. Used to calculate VaR(1-q), CoVaR(1-q)

        '''
                
        q = SetParams().q

        VaR = pd.DataFrame(index = [q, .99, .5], columns = dfLsim.columns)
        CoVaR = pd.DataFrame(index = [q, .99, .5], columns = dfLsim.columns)
        self.ECoVaR = pd.DataFrame(index = [q, .99,  .5], columns = dfLsim.columns)
        self.MES = pd.DataFrame(index = [q, .99,  .5], columns = dfLsim.columns)  
        self.ES = pd.DataFrame(index = [q, .99,  .5], columns = dfLsim.columns)
        self.ExSq = pd.DataFrame(index = dfLsim.columns, columns = dfLsim.columns) #Exposure Shortfall
        self.ExS99 = pd.DataFrame(index = dfLsim.columns, columns = dfLsim.columns) #Exposure Shortfall        

        VaR.loc[q] = dfLsim.quantile(q)
        VaR.loc[.5] = dfLsim.quantile(.5)
        VaR.loc[.99] = dfLsim.quantile(.99)
        
        self.Exptn = dfLsim.mean()
        
        self.MES.loc[q] = dfLsim[dfLsim['Sys']>=VaR.loc[q,'Sys']].mean()
        self.MES.loc[.99] = dfLsim[dfLsim['Sys']>=VaR.loc[.99,'Sys']].mean()
        self.MES.loc[.5] = dfLsim[dfLsim['Sys']>=VaR.loc[.5,'Sys']].mean()        
  
        'Calculate VaR, CoVaR, etc. for each entity'
        for jN, Name in enumerate(dfLsim.columns):
            dfLsimCoVaRq = dfLsim[dfLsim[Name]>=VaR.loc[q,Name]]
            CoVaR.loc[q,Name] = dfLsimCoVaRq.quantile(q)['Sys']
            dfLsimCoVaR5 = dfLsim[dfLsim[Name]>=VaR.loc[.5,Name]]            
            CoVaR.loc[.5,Name] = dfLsimCoVaR5.quantile(.5)['Sys']
            dfLsimCoVaR99 = dfLsim[dfLsim[Name]>=VaR.loc[.99,Name]]                        
            CoVaR.loc[.99,Name] = dfLsimCoVaR99.quantile(.99)['Sys']
            
            self.ES.loc[q,Name] = dfLsim[dfLsim[Name]>=VaR.loc[q,Name]][Name].mean()
            self.ES.loc[.99,Name] = dfLsim[dfLsim[Name]>=VaR.loc[.99,Name]][Name].mean()
            self.ES.loc[.5,Name] = dfLsim[dfLsim[Name]>=VaR.loc[.5,Name]][Name].mean()

            self.ExSq[Name] = dfLsim[dfLsim[Name] >= VaR.loc[q,Name]].mean()
            self.ExS99[Name] = dfLsim[dfLsim[Name] >= VaR.loc[.99,Name]].mean()
                            
        self.ECoVaR.loc[q] = dfLsim[dfLsim['Sys']>=VaR.loc[q,'Sys']].quantile(q)
        self.ECoVaR.loc[.5] = dfLsim[dfLsim['Sys']>=VaR.loc[.5,'Sys']].quantile(.5)
        self.ECoVaR.loc[.99] = dfLsim[dfLsim['Sys']>=VaR.loc[.99,'Sys']].quantile(.99)        
        
        self.VaR = VaR
        self.CoVaR = CoVaR
            
        self.DeltaCoVaR = self.CoVaR.loc[q] - self.CoVaR.loc[.5]
            
class DefaultP:
    def __init__(self,IndctrD, getCrossMatrix= False):
        '''
        + p_i_j : Probility of firm i defaulting conditional firm j defaulting
        + p_1, p_2, p_3 : Probability of at least 1, 2 or 3 firms defaulting 
        '''
        NSims = IndctrD.shape[0]
        universe = SetParams().universeFull
        NFirms = len(universe)
        
        'joint probability of distress : probab. that N or more will default'
        self.p = pd.DataFrame(np.nan, index = ['p1','p2','p3','p4'], columns =['p1ofN', 'pNafter1', 'pNafter2'])
        
        for jP in range(4):
            self.p.iloc[jP,0] = IndctrD['Sys'][IndctrD['Sys']>=jP+1].count() / NSims
            self.p.iloc[jP,1] = self.p.iloc[jP,0] /  self.p.iloc[0,0]
            if jP>1:
                self.p.iloc[jP,2] = self.p.iloc[jP,0] /  self.p.iloc[1,0]
        self.p.iloc[0,1] = np.NAN
        self.p.iloc[0,2] = np.NAN
        self.p.iloc[1,2] = np.NAN            
        'probability that at least one more will default, given that one defaults'
        
                            
        'conditional probab. of distress - loop through each firm conditional on each'
        self.cpd = pd.DataFrame(columns = universe) #expected number of defaults given that a firm defaults
        self.jpd = pd.DataFrame(columns = universe) #expected number of defaults given that a firm defaults
        self.sysIndex = pd.DataFrame(index = universe, columns =['POA','VI','SII'])            

        for jN, Name1 in enumerate(universe):
            'get systemic indicators based on PDs'                            
            CIndctrD = IndctrD[IndctrD[Name1]==1]            
            self.sysIndex.loc[Name1,'POA'] = 100*sum(IndctrD['Sys'][IndctrD[Name1]==1]>1)/sum(IndctrD[Name1]==1)
            self.sysIndex.loc[Name1,'SII'] = IndctrD['Sys'][IndctrD[Name1]>0].mean()
            self.sysIndex.loc[Name1,'VI'] = 100*IndctrD[Name1][IndctrD['Sys']>1].sum()/sum(IndctrD['Sys']>1)
                                    
            for jK, Name in enumerate(universe):
                if getCrossMatrix == True:
                    'Get the Defautl matrix :  All Names conditional on Name1 in distress'
                    for kN, Name2 in enumerate(universe):
                        'probab. that Name2 will be in distress conditional on Name1 in distress'
                        self.cpd.loc[Name1, Name2] = CIndctrD[Name2].sum()/CIndctrD[Name1].count()
                        'joint probability of distress of Name1 and Name2'
                        self.jpd.loc[Name1, Name2] = CIndctrD[Name2].sum()/NSims
        
            