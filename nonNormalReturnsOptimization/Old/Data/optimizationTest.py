import numpy as np
import pandas as pd
import math

from scipy.stats import norm
import time

import matplotlib.pyplot as plt

import yfinance as yf

payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
second_table = payload[1]

df = first_table
symbols = df['Symbol'].values.tolist()
names = df['Security'].values.tolist()


#show unique economy sectors
sectors = df['GICS Sector'].values.tolist()
sectors = set(sectors)

real_estate_df = df[df['GICS Sector'] == 'Real Estate']
real_estate_symbols = real_estate_df['Symbol'].values.tolist()


tickers = real_estate_symbols
num_stocks = len(tickers)

start = time.time()
data = yf.download(tickers, start='2015-08-01', end='2020-08-01', period = "1d")
price_data = data['Adj Close']
end = time.time()
print("Time: %0.2f seconds." % (end - start))


def convert_price_to_return(price_data):
    """
    Converts Price Data to Returns
    :param return_data: dataframe of returns of portfolio assets
    :return: returns minimum variance portfolio
    """
    portfolioAssets = price_data.head()

    return_data = pd.DataFrame()
    # Calculate Returns for Every Asset
    for asset in portfolioAssets:
        return_data[str(asset)] = (price_data[asset] - price_data[asset].shift(1)) / price_data[asset].shift(1)

    # Drop first row of Returns
    return_data = return_data.drop(return_data.index[0])
    return_data = return_data.dropna()
    return return_data


# Convert Price Data to Returns
return_data = convert_price_to_return(price_data)


def calculate_port_return(weights, mu):
    """
    Calculates the return of the portfolio
    :param weights: weight vector defining allocation between assets in portfolio
                mu: vector of historical returns of portfolio
    :return: returns return of portfolio
    """
    return np.dot(np.transpose(weights), mu)

def calculate_historical_returns(return_data):
    """
    Calculates the Mu matrix for the securities
    :param return_data: the data frame containing the returns
    :return: returns an array containing the arithmetic average return
    """
    return np.array(return_data.mean())

def calculate_port_variance(weights, Sigma):
    """
    Calculates the variance of returns of the portfolio
    :param weights: weight vector defining allocation between assets in portfolio
             sigma: covariance matrix of portfolio returns
    :return: returns variance of portfolio
    """
    return np.dot(np.dot(np.transpose(weights), Sigma), weights)

def calculate_value_at_risk(weights, return_data, alpha=0.95, lookback_days=520):
    # Calculate Weighted Portfolio Returns
    portfolio_returns = return_data.iloc[-lookback_days:].dot(weights)
    # Compute the correct percentile loss and multiply by value invested
    return np.percentile(portfolio_returns, 100 * (1-alpha))

def scale(x):
    return x / np.sum(np.abs(x))

weights = scale(np.random.random(num_stocks))

calculate_value_at_risk(weights, return_data, alpha=0.95)


lookback_days = 520
alpha = 0.95


def calculate_conditional_value_at_risk(weights, return_data, alpha=.95, lookback_days=520):
    # Call out to our existing function
    var = calculate_value_at_risk(weights, return_data, alpha=alpha, lookback_days=lookback_days)
    portfolio_returns = return_data.iloc[-lookback_days:].dot(weights)

    return np.nanmean(portfolio_returns[portfolio_returns < var])

lookback_days = 520
alpha = 0.95

# Multiply asset returns by weights to get one weighted portfolio return
portfolio_returns = return_data.iloc[-lookback_days:].dot(weights)

portfolio_VaR = calculate_value_at_risk(weights, return_data, alpha=alpha, lookback_days=lookback_days)

portfolio_CVaR = calculate_conditional_value_at_risk(weights, return_data, alpha=alpha, lookback_days=lookback_days)

# Plot only the observations > VaR on the main histogram so the plot comes out
# nicely and doesn't overlap.
plt.hist(portfolio_returns[portfolio_returns > portfolio_VaR], bins=20)
plt.hist(portfolio_returns[portfolio_returns < portfolio_VaR], bins=10)
plt.axvline(portfolio_VaR, color='red', linestyle='solid');
plt.axvline(portfolio_CVaR, color='red', linestyle='dashed');
plt.legend(['VaR for Specified Alpha as a Return',
            'CVaR for Specified Alpha as a Return',
            'Historical Returns Distribution',
            'Returns < VaR'])
plt.title('Historical VaR and CVaR');
plt.xlabel('Return');
plt.ylabel('Observation Frequency');
plt.show()


import scipy.optimize

def constraint_return_level(weights, return_data, return_level):
    """
    Constraint stating that the product of weights and returns should equal the mu null value
    :param weights: array of weights for the portfolio
    :param return_data: matrix of all the returns for all securities
    :param return_level: the specified return of the portfolio
    :return: returns the return of the portfolio minus the return level value
    """
    port_retrun = np.dot(np.transpose(weights), return_data)
    return port_retrun - return_level

def constraint_weight_sum(weights):
    """
    Constraint stating that all weights must sum to one
    :param w: the array of weights for the portfolio
    :return: returns the sum of weights minus one
    """
    return sum(weights) - 1

def efficient_weights_cvar(return_data, return_level, confidence, lookback_days=520):
    mu = return_data.mean()

    cons = [{'type': 'eq', 'fun': constraint_weight_sum},
            {'type': 'eq', 'fun': constraint_return_level, 'args': (mu, return_level)}]

    minimizer_kwargs = {"args": (return_data, confidence, lookback_days),
                        "constraints": cons}

    optimize = scipy.optimize.basinhopping(calculate_conditional_value_at_risk,
                                           x0=np.full((len(return_data.columns)), 1 / len(return_data.columns)),
                                           minimizer_kwargs=minimizer_kwargs)
    return optimize.x

return_level = .01
confidence = 0.95
eff_weights = efficient_weights_cvar(return_data, return_level, confidence)
