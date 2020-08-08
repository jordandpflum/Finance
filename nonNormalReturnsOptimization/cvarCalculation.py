from nonNormalReturnsOptimization.varCalculation import varCalculation
import numpy as np

def cvarCalculation(returnData, confidence):



    sortedReturns = returnData.sort_values(ascending=True)
    numReturns = len(sortedReturns)
    sortedReturnIndex = round((1-(100-confidence)*.01)*numReturns)
    cvar = (1/sortedReturnIndex)*sortedReturns[:sortedReturnIndex].sum()

    return cvar







