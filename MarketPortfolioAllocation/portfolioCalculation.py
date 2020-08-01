import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from prettytable import PrettyTable

priceData = pd.read_csv("Data/portfolioAdjPriceData.csv", index_col=0)

numAssets = len(priceData.columns)

returnData = pd.DataFrame()
# Calculate Returns for Every Asset
for asset in priceData.head():
    returnData[str(asset + "_returns")] = (priceData[asset] - priceData[asset].shift(1)) / priceData[asset].shift(1)

# Drop first row of Returns
returnData = returnData.drop(returnData.index[0])

# Calculate Historical Returns
historicalReturns = returnData.mean()

# Convert Returns to Array
historicalReturns = historicalReturns.to_numpy()

# Calculate Risk
risk = returnData.std().to_numpy()

covMatrix = returnData.cov().to_numpy()

# Define RF (monthly)
irx = 0.98
rf = irx*.01/12


# GMV Calculation
# Create b
b = np.zeros( (numAssets+1, 1) )
# Set Initial Constraint (w'1 = 1)
b[numAssets, 0] = 1

# Create A
A = np.zeros( (numAssets+1, numAssets+1) )

A[0:numAssets, 0:numAssets] = 2*covMatrix
A[numAssets,0:numAssets] = np.ones( (1,numAssets) )
A[0:numAssets,numAssets] = np.ones( (1,numAssets) )

# Solve for x
x = np.dot(np.linalg.inv(A), b)


# Calculate Minimum Variance Portfolio
w = x[0:numAssets]
erp = np.dot(np.transpose(w), historicalReturns)
sigmarp = math.sqrt(np.dot(np.transpose(w), np.dot(covMatrix, w)))
minVarPort = (float(erp), float(sigmarp))

# MVE Calculation
# Create A
A = np.zeros( (numAssets+2, numAssets+2) )

A[0:numAssets, 0:numAssets] = 2*covMatrix
A[numAssets,0:numAssets] = np.ones( (1,numAssets) )
A[0:numAssets,numAssets] = np.ones( (1,numAssets) )
A[numAssets+1,0:numAssets] = historicalReturns
A[0:numAssets,numAssets+1] = historicalReturns

EF = [[],[]]

mu0Increment = .001
mu0Itterations = 25

for mu0 in np.arange(minVarPort[0], mu0Itterations*mu0Increment + minVarPort[0], mu0Increment):
    # Create b
    b = np.zeros( (numAssets+2, 1) )
    # Set Initial Constraints (w'1 = 1), (E(rp)=mu0)
    b[numAssets, 0] = 1
    b[numAssets+1, 0] = mu0

    # Solve for x
    x = np.dot(np.linalg.inv(A), b)

    # Calculate EF Portfolio
    w = x[0:numAssets]
    erp = np.dot(np.transpose(w), historicalReturns)
    sigmarp = math.sqrt(np.dot(np.transpose(w), np.dot(covMatrix, w)))
    EF[0].append(erp)
    EF[1].append(sigmarp)



# ORP Calculation
# Weight Vector
w = np.dot(np.linalg.inv(covMatrix), np.reshape(historicalReturns, (numAssets,1)) - np.dot(rf, np.ones( (numAssets, 1)))) / \
    np.dot(np.dot(np.transpose(np.ones( (numAssets, 1))), np.linalg.inv(covMatrix)),np.reshape(historicalReturns, (numAssets,1)) - np.dot(rf, np.ones( (numAssets, 1))))

# Calculate Optimal Risky Portfolio
erp = np.dot(np.transpose(w), historicalReturns)
sigmarp = math.sqrt(np.dot(np.transpose(w), np.dot(covMatrix, w)))
ORPort = (float(erp), float(sigmarp))

# CAL Line
optimalSharpeRatio = float((ORPort[0] - rf) / ORPort[1])
calRisks = np.arange(0, EF[1][-1], mu0Increment)
calErps = list(map(lambda x: x * optimalSharpeRatio + rf, np.arange(0, EF[1][-1], mu0Increment)))


# Random Portfolio
randPort_w = np.array([[0.9],
                         [0.05],
                         [0],
                         [0],
                         [0],
                         [0],
                         [0.05]]
                        )
erp = np.dot(np.transpose(randPort_w), historicalReturns)
sigmarp = math.sqrt(np.dot(np.transpose(randPort_w), np.dot(covMatrix, randPort_w)))
randPort = (float(erp), float(sigmarp))

# Calculate Efficient Random Portfolio (same return as random)
# Create b
b = np.zeros((numAssets + 2, 1))
# Set Initial Constraints (w'1 = 1), (E(rp)=mu0)
b[numAssets, 0] = 1
b[numAssets + 1, 0] = randPort[0]

# Solve for x
x = np.dot(np.linalg.inv(A), b)

# Calculate EF Portfolio
w = x[0:numAssets]
erp = np.dot(np.transpose(w), historicalReturns)
sigmarp = math.sqrt(np.dot(np.transpose(w), np.dot(covMatrix, w)))
randPortEff = (float(erp), float(sigmarp))

# Calculate Complete Random Portfolio (same return as original random portfolio)
erp = randPort[0]
sigmarp = (randPort[0] - rf)/optimalSharpeRatio
randPortComplete_ESame = (float(erp), float(sigmarp))


# Calculate Complete Random Portfolio (same risk as original random portfolio)
erp = optimalSharpeRatio*randPort[1] + rf
sigmarp = randPort[1]
randPortComplete_RiskSame = (float(erp), float(sigmarp))


# Summary Table
table = PrettyTable(['Portfolio', 'Annual Returns', 'Annual SD', 'Annual Returns Diff to Random Portfolio', 'Annual SD Diff to Random Portfolio'])
portfolios = [["Random Portfolio", randPort],
              ["Efficient Portfolio (Same Returns as Random Portfolio)", randPortEff],
              ["Optimal Portfolio", ORPort],
              ["Optimal Complete Portfolio (Same Returns as Random Portfolio)",randPortComplete_ESame],
              ["Optimal Complete Portfolio (Same Risk as Random Portfolio)", randPortComplete_RiskSame]
              ]
originalPort_AnnualRet = (1+randPort[0])**12 - 1
originalPort_AnnualSD = randPort[1] * math.sqrt(12)
for portfolio in portfolios:
    annual_ret = (1+portfolio[1][0])**12 - 1
    annual_sd = portfolio[1][1] * math.sqrt(12)
    rec = [portfolio[0], annual_ret, annual_sd, annual_ret-originalPort_AnnualRet, annual_sd-originalPort_AnnualSD]
    table.add_row(rec)

print(table)

# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(111)

#ax1.scatter(EF[1], EF[0], s=10, c='b', marker="s", label='EF')
plt.plot(EF[1], EF[0], linestyle='-', c='b', marker="s", label='EF')
ax1.scatter(ORPort[1], ORPort[0], s=10, c='r', marker="o", label='Optimal Risky Portfolio')
ax1.scatter(randPort[1], randPort[0], s=10, c='g', marker="o", label='Random Portfolio')
ax1.scatter(randPortEff[1], randPortEff[0], s=10, c='y', marker="o", label='Random Efficient Portfolio')
ax1.scatter(randPortComplete_ESame[1], randPortComplete_ESame[0], s=10, c='k', marker="o", label='Complete Random Portfolio (Same Return)')
ax1.scatter(randPortComplete_RiskSame[1], randPortComplete_RiskSame[0], s=10, c='m', marker="o", label='Complete Random Portfolio (Same Risk)')
plt.plot(calRisks, calErps, label='CAL')
plt.legend(loc='upper left', prop={'size': 6});
plt.title('Market Portfolio')
plt.xlabel('sigma')
plt.ylabel('E(rp)')
#plt.savefig('marketPortfolioOptimization.png')
plt.show()


x_1 = np.linspace(randPort[0] - 3*randPort[1], randPort[0] + 3*randPort[1], 100)
x_2 = np.linspace(randPortComplete_RiskSame[0] - 3*randPortComplete_RiskSame[1], randPortComplete_RiskSame[0] + 3*randPortComplete_RiskSame[1], 100)
plt.plot(x_1, stats.norm.pdf(x_1, randPort[0], randPort[1]), label='Random Portfolio')
plt.plot(x_2, stats.norm.pdf(x_2, randPortComplete_RiskSame[0], randPortComplete_RiskSame[1]), label='Complete Random Portfolio (Same Risk)')
plt.legend(loc='upper left', prop={'size': 5});
plt.title('Portfolio Distributions')
plt.xlabel('E(rp)')
plt.show()

x_1 = np.linspace(randPort[0] - 3*randPort[1], randPort[0] + 3*randPort[1], 100)
x_2 = np.linspace(randPortComplete_ESame[0] - 3*randPortComplete_ESame[1], randPortComplete_ESame[0] + 3*randPortComplete_ESame[1], 100)
plt.plot(x_1, stats.norm.pdf(x_1, randPort[0], randPort[1]), label='Random Portfolio')
plt.plot(x_2, stats.norm.pdf(x_2, randPortComplete_ESame[0], randPortComplete_ESame[1]), label='Complete Random Portfolio (Same Risk)')
plt.legend(loc='upper left', prop={'size': 5});
plt.title('Portfolio Distributions')
plt.xlabel('E(rp)')
plt.show()