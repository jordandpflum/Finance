import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import random


import timeit


start = timeit.default_timer()
# Read in Data
return_data = pd.read_csv('Data/returnData_100Assets.csv', index_col=0, parse_dates=True)
market_data = pd.read_csv('Data/returnData.csv', index_col=0, parse_dates=True)

# Subset reutrn to Fit in Time Range of Market Data
start_date = market_data.index[0]
end_date = market_data.index[-1]
return_data = return_data[start_date:end_date]


# Get sample portfolio of n assets
num_assets = 10
tickers_list = random.sample(list(return_data.columns), num_assets)  # careful tweaking this number. This becomes computationally expensive
n_port_stdev = [[],[]]

def calculate_portfolio_variance(tickers, data):
    num_securities = len(tickers)
    # Equally weighted portfolios
    security_weights = 1 / num_securities
    m_weight = np.full((num_securities, 1), security_weights)
    m_cov = data[tickers].cov()
    portfolio_variance = np.dot(np.dot(np.transpose(m_weight), m_cov), m_weight)
    return portfolio_variance

itt_count = 0
for L in range(1, len(tickers_list) + 1):
    port_variances_at_n = []
    for subset in itertools.combinations(tickers_list, L):
        l_subset = list(subset)
        port_val = calculate_portfolio_variance(l_subset, return_data)
        port_variances_at_n.append(port_val)
        itt_count += 1


    n_port_stdev[0].append(L)
    n_port_stdev[1].append(math.sqrt(float(sum(port_variances_at_n) / len(port_variances_at_n))) * math.sqrt(12))
    #n_port_stdev[L] = math.sqrt(float(sum(port_variances_at_n) / len(port_variances_at_n))) * math.sqrt(12)
    #n_port_stdev[L] = math.sqrt(float(sum(port_variances_at_n) / len(port_variances_at_n))) * math.sqrt(12)

print(itt_count)
stop = timeit.default_timer()
print('Time: ', stop - start)

print(tickers_list)
print(n_port_stdev)
var_spy = (np.std(market_data["SPY"]) * np.sqrt(12))**2
plt.plot(range(1, len(tickers_list) + 1), np.full((len(tickers_list)), var_spy))
plt.plot(n_port_stdev[0], n_port_stdev[1])
plt.xlabel('number of securities')
plt.ylabel('average annual portfolio risk')
plt.show()


