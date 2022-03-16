''' Several useful plot style appropriate for academic papers
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import numpy as np


def myplot_frame(figSize=(4, 3)):
    'plot function parameters'
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(1, 1, 1)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')

    return ax

def myplot(x,y,labels, lineStyle='-'):
    '''
    lineStyle : 'dashed' or 'solid'
    labels : a collection two strings 'string', xlabel and ylabel
    title: a 'string'
    '''
    xlabel, ylabel = labels #, lineLabel
    ax = myplot_frame()
    ax.plot(x, y,  color='k', ls=lineStyle) #, label = r'$t^*$' , label=lineLabel 
    ax.set_xlabel(xlabel,fontsize='xx-large')
    ax.set_ylabel(ylabel, fontsize='xx-large')
    #ax.legend()

    return ax

def lollipopChart(df,myColor='grey'):
    #fig, ax = plt.subplots(figsize=(6,6), dpi= 80)
    '''
    ax : either pre-specified plot axis or False otherwise
    color : aternatively use firebrick
    '''
    ax = myplot_frame()
    y = df.columns[0]
    df.sort_values(by=y, ascending=False, inplace=True)
    ax.vlines(x=df.index, ymin=0, ymax=df[y], color=myColor, alpha=0.4, linewidth=2)
    ax.scatter(x=df.index, y=df[y], s=75, color='navy', alpha=0.7)
    
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index.str.upper(), rotation=60)
    ax.set_ylim(0)
    
    # Annotate
    for row in df.itertuples():
        ax.text(row.Index, row[1], s=round(row[1], 2), horizontalalignment= 'center', verticalalignment='bottom')
    
    return ax

def lollipopChart2(df):
    'lollipop plot of 2 series from a dataFrame'
    ax = myplot_frame((7,6))
    y1 = df.columns[0]
    y2 = df.columns[1]
    df.sort_values(by=y1, ascending=False, inplace=True)
      
    ax.vlines(x=df.index, ymin=df[y2], ymax=df[y1], color='grey', alpha=0.4)
    ax.scatter(df.index, df[y1], color='navy', alpha=1, label=y1)
    ax.scatter(df.index, df[y2], color='gold', alpha=0.8 , label=y2)
    #ax.set_xticks(df.index)
    ax.set_xticklabels(df.index.str.upper())
    ax.legend()
    
    for row in df[y1].to_frame().itertuples():
        ax.text(row.Index, row[1], s=round(row[1], 2), horizontalalignment= 'center', verticalalignment='bottom')

    for row in df[y2].to_frame().itertuples():
        ax.text(row.Index, row[1], s=round(row[1], 2), horizontalalignment= 'center', verticalalignment='bottom')

    return ax

def myHeatmapDoubleM(df):
    'Heatmap matrix, carving out the diagonal'
    ax= myplot_frame((7,6)) #plt.subplot(111, polar=True)    
    mask = np.invert(np.ones_like(df, dtype=np.bool))    
    np.fill_diagonal(mask,True)
    sns.heatmap(df.round(2), mask=mask, annot=True, fmt=".3g", cmap='Oranges', vmin=0, vmax=df[np.invert(mask)].max().max(), ax=ax)
    return ax
