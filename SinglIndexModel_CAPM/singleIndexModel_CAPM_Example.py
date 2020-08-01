import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import math
import scipy.stats as stats
from prettytable import PrettyTable



# Read in Data
returnData = pd.read_csv('Data/returnData.csv', index_col=0)
marketProxy = returnData["SPY"]
portfolio = returnData.drop("SPY", axis=1)

tickers = list(portfolio.columns)
numAssets = len(portfolio.columns)

# Define RF (monthly)
irx = 0.98
rf = irx*.01/12

# SCL
# Calculate Risk Premium
portfolioRiskPremium = portfolio.apply(lambda x: x - rf)
marketRiskPremium = marketProxy.apply(lambda x: x - rf)

marketVariance = np.var(marketProxy)

SCL = pd.DataFrame(index=tickers,columns=["Beta", "Alpha", "Total Risk", "Systematic Risk", "Non-Sys Risk"])
for ticker in tickers:
    # Calculate Beta and Alpha
    beta, alpha = np.polyfit(marketRiskPremium, portfolioRiskPremium[ticker], 1)
    SCL.loc[ticker, 'Beta'] = beta
    SCL.loc[ticker, 'Alpha'] = alpha
    SCL.loc[ticker, 'Total Risk'] = np.var(portfolioRiskPremium[ticker])
    SCL.loc[ticker, 'Systematic Risk'] = beta**2 * marketVariance
    SCL.loc[ticker, 'Non-Sys Risk'] = np.var(portfolioRiskPremium[ticker]) - beta**2 * marketVariance

# SML
slope = np.average(marketRiskPremium)
eri = list(map(lambda x: x * slope + rf, SCL["Beta"]))

# Plotting
# SCL
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(marketRiskPremium, portfolioRiskPremium["AAPL"], s=10, c='r', marker="o")
plt.plot(marketRiskPremium, beta*marketRiskPremium + alpha)
#plt.legend(loc='upper left', prop={'size': 6});
plt.title('SCL')
plt.xlabel('rm - rf')
plt.ylabel('ri - rf')
#plt.savefig('marketPortfolioOptimization.png')
plt.show()

# SML
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(SCL["Beta"], eri, s=10, c='r', marker="o")
'''
# Adding Line Segments
lines = [[(0, np.average(marketRiskPremium)), (1, np.average(marketRiskPremium))],
         [(1, 0), (1, np.average(marketRiskPremium))]
         ]
c = np.array([(0, 0, 1, 1), (0, 0, 1, 1)])
lc = mc.LineCollection(lines, colors=c, linewidths=2)
ax1.add_collection(lc)
ax1.autoscale()
ax1.margins(0)
'''

#plt.plot(marketRiskPremium, beta*marketRiskPremium + alpha)
#plt.legend(loc='upper left', prop={'size': 6});

plt.title('SML')
plt.xlabel('Beta')
plt.ylabel('E(ri)')
#plt.savefig('marketPortfolioOptimization.png')
plt.show()


#ax1.autoscale()
#ax1.margins(0.1)

