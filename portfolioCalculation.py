import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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
rf = irx*.1*.01/12


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
mu0Itterations = 20

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



# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(EF[1], EF[0], s=10, c='b', marker="s", label='EF')
ax1.scatter(ORPort[1], ORPort[0], s=10, c='r', marker="o", label='Optimal Risky Portfolio')
plt.plot(calRisks, calErps, label='CAL')
plt.legend(loc='upper left');
plt.title('Market Portfolio')
plt.xlabel('sigma')
plt.ylabel('E(rp)')
plt.show()