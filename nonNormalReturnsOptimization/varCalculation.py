

import numpy as np
import math

def varCalculation(returnData, confidence=5):

    sortedReturns = returnData.sort_values(ascending=True)
    VaR = np.percentile(sortedReturns, confidence)

    return VaR





