import pandas as pd
from nonNormalReturnsOptimization.portfolioCalculation import *



priceData = pd.read_csv("Data/portfolioAdjPriceData.csv", index_col=0)

# Set Number of Assets
numAssets = len(priceData.columns)

# Convert to Return Data
returnData = convertPriceDataToReturns(priceData)


# Calculate Preliminary Values
[historicalReturns, risk, covMatrix, corrMatrix] = calculatePreliminaryValues(returnData)

# Define RF (monthly)
irx = 0.98
rf = irx*.01/12


# Calculate Minimum Variance Portfolio
minVarPort = calculateGMV(numAssets, covMatrix, historicalReturns, annualized=False)

# Efficient Frontier
mu0Increment = .001
mu0Itterations = 25
EF = calculateEF(numAssets, covMatrix, historicalReturns,
                 minVarPort, mu0Increment, mu0Itterations)

# Optimal Risky Portfolio
ORPort = calculateORP(numAssets, covMatrix, historicalReturns, rf, annualized=False)


# CAL
[calErps, calRisks] = calculateCAL(ORPort, rf, EF, mu0Increment)


# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(111)

#ax1.scatter(EF[1], EF[0], s=10, c='b', marker="s", label='EF')
plt.plot(EF[1], EF[0], linestyle='-', c='b', marker="s", label='EF')
ax1.scatter(ORPort[1], ORPort[0], s=10, c='r', marker="o", label='Optimal Risky Portfolio')
#ax1.scatter(randPort[1], randPort[0], s=10, c='g', marker="o", label='Random Portfolio')
#ax1.scatter(randPortEff[1], randPortEff[0], s=10, c='y', marker="o", label='Random Efficient Portfolio')
#ax1.scatter(randPortComplete_ESame[1], randPortComplete_ESame[0], s=10, c='k', marker="o", label='Complete Random Portfolio (Same Return)')
#ax1.scatter(randPortComplete_RiskSame[1], randPortComplete_RiskSame[0], s=10, c='m', marker="o", label='Complete Random Portfolio (Same Risk)')
plt.plot(calRisks, calErps, label='CAL')
plt.legend(loc='upper left', prop={'size': 6});
plt.title('Market Portfolio')
plt.xlabel('sigma')
plt.ylabel('E(rp)')
#plt.savefig('marketPortfolioOptimization.png')
plt.show()









