import pandas as pd
from scipy.stats import norm
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
#plt.show()


from nonNormalReturnsOptimization.varCalculation import varCalculation
from nonNormalReturnsOptimization.cvarCalculation import cvarCalculation

exampleAsset = "SPY_AdjClose_returns"
confidence=5
var = varCalculation(returnData[exampleAsset], confidence)
cvar = cvarCalculation(returnData[exampleAsset], confidence)

print("5.0% VaR threshold:",var)
print("5.0% CVaR:",cvar)


# Probabilistic Density Function
mu = returnData[exampleAsset].mean()
sigma = returnData[exampleAsset].std()
kurtosis = returnData[exampleAsset].kurtosis()
skewness = returnData[exampleAsset].skew()
num_bins = int(np.sqrt(len(returnData[exampleAsset])))

print("Kurtosis:",kurtosis)
print("skewness:",skewness)
fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(returnData[exampleAsset], num_bins, density=1,color="#002080",label='',alpha=0.95)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)

plt.xlabel('Returns')
plt.xticks(rotation=45)
plt.yticks()
plt.ylabel('Probability density')
ax.set_title("VaR for SPY \n Mu = %.3f, Sigma = %.3f, Kurtosis = %.3f \n" % (mu,sigma,kurtosis), fontsize=12)
plt.axvline(x=var, color='#ff4d4d', linestyle='--',linewidth=2, label='VaR at {}%: '.format(str(float(100-confidence))) + str(round(var,3)))
plt.axvline(x=cvar, color='r', linestyle='-',linewidth=2, label='CVaR at {}% VaR: '.format(str(float(100-confidence))) + str(round(cvar,3)))
plt.axvline(x=mu, color='w', linestyle='dashed',linewidth=2, label = 'Mean: ' + str(round(mu,3)))
ax.set_facecolor('w')
plt.legend(loc="upper right")
plt.show()
#plt.savefig('SPY Var.png', dpi=300,bbox_inches='tight')



from scipy.stats import norm
def calculate_VaR(returns, alpha):
    """
    Calculates the value at risk at a given alpha level
    :param returns: data frame of returns
    :param alpha: alpha level
    :return: Value at Risk
    """
    stdev = returns.std()
    mean = returns.mean()
    VaR = -(norm.ppf(1 - alpha, loc=mean, scale=stdev))
    print(VaR)
    return VaR
def calculate_CVaR(returns, alpha, VaR):
    """
    Calculates the conditional value at risk at a given alpha level
    :param returns: data frame of returns
    :param alpha: onfidence level
    :param VaR: specified value at risk
    :return: Conditional Value at Risk
    """
    stdev = returns.std()
    mean = returns.mean()
    pdf_at_VaR = norm(loc=mean, scale=stdev).pdf(1 - alpha)
    CVaR = (0 - VaR * pdf_at_VaR) * (1 / (1 - alpha))
    return CVaR

var = calculate_VaR(returnData[exampleAsset], .05)
print("5.0% VaR threshold:",var)
cvar = calculate_CVaR(returnData[exampleAsset], .05, var)

print("5.0% VaR threshold:",var)
print("5.0% CVaR:",cvar)





