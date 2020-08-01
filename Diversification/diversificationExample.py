import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import random



# Read in Data
price_data = pd.read_csv('Data/returnData.csv', index_col=0)


# Get sample portfolio of n assets
num_assets = 7
tickers_list = random.sample(list(price_data.columns), num_assets)  # careful tweaking this number. This becomes computationally expensive
n_port_stdev = [[],[]]

def calculate_portfolio_variance(tickers, data):
    num_securities = len(tickers)
    # Equally weighted portfolios
    security_weights = 1 / num_securities
    m_weight = np.full((num_securities, 1), security_weights)
    m_cov = data[tickers].cov()
    portfolio_variance = np.dot(np.dot(np.transpose(m_weight), m_cov), m_weight)
    return portfolio_variance


for L in range(1, len(tickers_list) + 1):
    port_variances_at_n = []
    for subset in itertools.combinations(tickers_list, L):
        l_subset = list(subset)
        port_val = calculate_portfolio_variance(l_subset, price_data)
        port_variances_at_n.append(port_val)

    n_port_stdev[0].append(L)
    n_port_stdev[1].append(math.sqrt(float(sum(port_variances_at_n) / len(port_variances_at_n))) * math.sqrt(12))
    #n_port_stdev[L] = math.sqrt(float(sum(port_variances_at_n) / len(port_variances_at_n))) * math.sqrt(12)
    #n_port_stdev[L] = math.sqrt(float(sum(port_variances_at_n) / len(port_variances_at_n))) * math.sqrt(12)

print(tickers_list)
print(n_port_stdev)
var_spy = np.std(price_data["SPY"])
plt.plot(range(1, len(tickers_list) + 1), np.full((len(tickers_list)), var_spy))
plt.plot(n_port_stdev[0], n_port_stdev[1])
#plt.xlabel('number of securities')
#plt.ylabel('average annual portfolio risk')
plt.show()


