import scipy.optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


pd.set_option('display.max_columns', None)
pd.options.display.width = 0

df_returns = pd.read_csv('Data2.csv', index_col=0)


def calculate_mu(return_data):
    """
    Calculates the Mu matrix for the securities
    :param return_data: the data frame containing the returns
    :return: returns an array containing the arithmetic average return
    """
    return np.array(return_data.mean())


def calculate_cvar(arguments, returns, confidence):
    """
    Calculates the conditional value at risk at a given confidence level
    :param arguments: the iterative weights thrown in by the optimizer
    :param returns: data frame of returns for all securities
    :param confidence: confidence level for value at risk
    :return: the conditional value at risk at a given confidence level at the portfolio level
    """

    w = np.array(arguments)
    portfolio_returns = np.dot(returns, w)
    portfolio_returns_sorted = np.sort(portfolio_returns, axis=0)
    n_returns = len(portfolio_returns_sorted)
    confidence_index = round((1 - (100 - confidence) * 0.01) * n_returns)
    c_var = (1 / confidence_index) * portfolio_returns_sorted[:confidence_index].sum()
    return c_var


def constraint_sum(w):
    """
    Constraint stating that all weights must sum to one which is used by the optimizer
    :param w: the array of weights for the portfolio
    :return: returns the sum of weights minus one
    """
    return sum(w) - 1


def constraint_mu_null(w, mu, mu_null):
    """
    Constraint stating that the product of weights and returns should equal the mu null value
    :param w: array of weights for the portfolio
    :param mu: matrix of all the returns for all securities
    :param mu_null: the specified mu null value
    :return: returns the return of the portfolio minus the mu null value
    """
    ret = np.dot(np.transpose(w), mu)
    return ret - mu_null


plot_data = {}


total_portfolios = 10
for i in range(total_portfolios):
    sys.stdout.write('\r')
    sys.stdout.write('Percent Complete:  ' + str(round((i / total_portfolios) * 100, 2)) + '%')
    sys.stdout.flush()

    confidence_val = 95
    mu_null_value = 0.001 * i + 0.001
    mu_matrix = df_returns.mean()
    covariance_matrix = df_returns.cov()
    cons = [{'type': 'eq', 'fun': constraint_sum},
            {'type': 'eq', 'fun': constraint_mu_null, 'args': (mu_matrix, mu_null_value,)}]

    optimize = scipy.optimize.minimize(calculate_cvar,
                                       x0=np.full((len(df_returns.columns)), (1 / (len(df_returns.columns)))),
                                       args=(df_returns, confidence_val),
                                       constraints=cons,
                                       options={'maxiter': 20000})

    plot_data[optimize.fun] = mu_null_value


plt.plot(plot_data.keys(), plot_data.values())
plt.show()

