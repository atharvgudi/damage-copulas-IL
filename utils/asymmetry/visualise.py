import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .matrices import *

def plotAsymmetry(df, asymmetry='concordance'):
    '''
        FUNCTION
        --------
        Plots pairwise asymmetries against rank correlations for all variable pairs,


        PARAMETERS
        ----------
        df:         DataFrame with observations as rows and variables as columns
                    REQ: pandas.DataFrame

        asymmetry:  Type of asymmetry to compute and test
                    REQ: String of either 'concordance' or 'direct'


        RETURNS
        -------
        list[(str, str)]
        sigPairs:    Variable name pairs whose asymmetry is statistically significant
    '''

    '''
    Mapping to determine which asymmetry matrix + confidence interval function to use 
        based on the asymmetry type in function signature
    '''
    asymFns    = {'concordance': concAsyMatrix,  'direct': dirAsyMatrix}
    confIntFns = {'concordance': concAsyConfInt, 'direct': dirAsyConfInt}

    if asymmetry not in asymFns:
        raise ValueError(f"asymmetry must be one of {list(asymFns.keys())}")

    data = df.values
    cols  = list(df.columns)
    nDim = data.shape[1]
    n    = data.shape[0]

    rCorrMatrix              = rankCorrMatrix(data)
    nRCorrMatrix             = normRankCorrMatrix(data)
    asyMatrix                = asymFns[asymmetry](data)
    lowerBounds, upperBounds = confIntFns[asymmetry](nRCorrMatrix, n)

    ranks,    asymmetries    = [], []
    sigRanks, sigAsymmetries = [], []
    sigPairs, sigLabels      = [], []

    for var1 in range(nDim - 1):
        for var2 in range(var1 + 1, nDim):

            ranks      += [rCorrMatrix[var1, var2]]
            asymValue  = asyMatrix[var1, var2]
            asymmetries += [asymValue]

            lower = lowerBounds[var1, var2]
            upper = upperBounds[var1, var2]

            if (asymValue > upper) or (asymValue < lower):
                sigRanks       += [rCorrMatrix[var1, var2]]
                sigAsymmetries += [asymValue]
                sigPairs       += [(cols[var1], cols[var2])]
                sigLabels      += [cols[var1] + ' & ' + cols[var2]]

    plt.scatter(ranks, asymmetries)
    for i in range(len(sigRanks)):
        plt.scatter(sigRanks[i], sigAsymmetries[i],
                    marker='+', s=154, label=sigLabels[i])
    plt.title("Rank correlations and asymmetries for all pairs")
    plt.xlabel("Rank Correlation", fontweight='bold')
    plt.ylabel("Asymmetry",        fontweight='bold')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=1, fancybox=True, shadow=True)

    plt.show()

    return sigPairs

