import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from prettytable import PrettyTable


def convertPriceDataToReturns(priceData):

    # Create Empty Dataframe
    returnData = pd.DataFrame()

    # Calculate Returns for Every Asset
    for asset in priceData.head():
        returnData[str(asset + "_returns")] = (priceData[asset] - priceData[asset].shift(1)) / priceData[asset].shift(1)

    # Drop first row of Returns
    returnData = returnData.drop(returnData.index[0])

    return returnData


def calculatePreliminaryValues(returnData):

    # Calculate Historical Returns
    historicalReturns = returnData.mean().to_numpy()

    # Calculate Historical Risk
    risk = returnData.std().to_numpy()

    # Calculate Covariance Matrix
    covMatrix = returnData.cov().to_numpy()

    # Calculate Correlation Matrix
    corrMatrix = returnData.corr().to_numpy()

    return historicalReturns, risk, covMatrix, corrMatrix

def calculateGMV(numAssets, covMatrix, historicalReturns, annualized=False, periodRate = 12):
    # Define b matrix
    b = np.zeros((numAssets + 1, 1))
    # Set Initial Constraint (w'1 = 1)
    b[numAssets, 0] = 1

    # Create A
    A = np.zeros((numAssets + 1, numAssets + 1))
    # Fill A
    A[0:numAssets, 0:numAssets] = 2 * covMatrix
    A[numAssets, 0:numAssets] = np.ones((1, numAssets))
    A[0:numAssets, numAssets] = np.ones((1, numAssets))

    # Solve for x
    x = np.dot(np.linalg.inv(A), b)

    # Calculate Minimum Variance Portfolio
    w = x[0:numAssets]
    erp = np.dot(np.transpose(w), historicalReturns)
    sigmarp = math.sqrt(np.dot(np.transpose(w), np.dot(covMatrix, w)))
    minVarPort = (float(erp), float(sigmarp))

    if annualized:
        erp = (1+erp)**periodRate - 1
        sigmarp = sigmarp * math.sqrt(sigmarp)
        minVarPort = (float(erp), float(sigmarp))

    return minVarPort


def calculateMVE(numAssets, covMatrix, historicalReturns, mu0, annualized=False, periodRate = 12):
    # Create A
    A = np.zeros((numAssets + 2, numAssets + 2))

    # Fill A
    A[0:numAssets, 0:numAssets] = 2 * covMatrix
    A[numAssets, 0:numAssets] = np.ones((1, numAssets))
    A[0:numAssets, numAssets] = np.ones((1, numAssets))
    A[numAssets + 1, 0:numAssets] = historicalReturns
    A[0:numAssets, numAssets + 1] = historicalReturns

    # Create b
    b = np.zeros((numAssets + 2, 1))

    # Set Initial Constraints (w'1 = 1), (E(rp)=mu0)
    b[numAssets, 0] = 1
    b[numAssets + 1, 0] = mu0

    # Solve for x
    x = np.dot(np.linalg.inv(A), b)

    # Calculate EF Portfolio
    w = x[0:numAssets]
    erp = np.dot(np.transpose(w), historicalReturns)
    sigmarp = math.sqrt(np.dot(np.transpose(w), np.dot(covMatrix, w)))
    MVEPortfolio = (float(erp), float(sigmarp))

    if annualized:
        erp = (1+erp)**periodRate - 1
        sigmarp = sigmarp * math.sqrt(sigmarp)
        MVEPortfolio = (float(erp), float(sigmarp))

    return MVEPortfolio

def calculateEF(numAssets, covMatrix, historicalReturns, minVarPort, mu0Increment, mu0Itterations, annualized=False, periodRate = 12):
    EF = [[], []]
    for mu0 in np.arange(minVarPort[0], mu0Itterations * mu0Increment + minVarPort[0], mu0Increment):
        MVEPortfolio = calculateMVE(numAssets, covMatrix, historicalReturns, mu0, annualized=annualized, periodRate=periodRate)
        EF[0].append(MVEPortfolio[0])
        EF[1].append(MVEPortfolio[1])

    return EF


def calculateORP(numAssets, covMatrix, historicalReturns, rf, annualized=False, periodRate = 12):
    # Weight Vector
    w = np.dot(np.linalg.inv(covMatrix),
               np.reshape(historicalReturns, (numAssets, 1)) - np.dot(rf, np.ones((numAssets, 1)))) / \
        np.dot(np.dot(np.transpose(np.ones((numAssets, 1))), np.linalg.inv(covMatrix)),
               np.reshape(historicalReturns, (numAssets, 1)) - np.dot(rf, np.ones((numAssets, 1))))

    # Calculate Optimal Risky Portfolio
    erp = np.dot(np.transpose(w), historicalReturns)
    sigmarp = math.sqrt(np.dot(np.transpose(w), np.dot(covMatrix, w)))
    ORPort = (float(erp), float(sigmarp))

    if annualized:
        erp = (1+erp)**periodRate - 1
        sigmarp = sigmarp * math.sqrt(sigmarp)
        ORPort = (float(erp), float(sigmarp))

    return ORPort


def calculateCAL(ORPort, rf, EF, mu0Increment, annualized=False, periodRate=12):
    optimalSharpeRatio = float((ORPort[0] - rf) / ORPort[1])
    calRisks = np.arange(0, EF[1][-1], mu0Increment)
    calErps = list(map(lambda x: x * optimalSharpeRatio + rf, np.arange(0, EF[1][-1], mu0Increment)))

    if annualized:
        calRisks = list(map(lambda x: x*math.sqrt(periodRate), calRisks))
        calErps = list(map(lambda x: (1+x)**periodRate - 1, calErps))

    return calErps, calRisks



