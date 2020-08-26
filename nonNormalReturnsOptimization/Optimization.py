import scipy.optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
from scipy.stats import norm
from scipy.stats import t



pd.set_option('display.max_columns', None)
pd.options.display.width = 0

df_returns_data = pd.read_csv('C:\\Users\\jason\\PycharmProjects\\UT MSBA\\Decision Analytics\\Data.csv', index_col=0)
df_returns_hf = pd.read_csv('Data_HF - Copy.csv', index_col=0).dropna()
# print(df_returns_hf)

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
    confidence_index = round(((100 - confidence) * 0.01) * n_returns)
    # confidence_index = round((1 - (100 - confidence) * 0.01) * n_returns)
    c_var = (1 / confidence_index) * portfolio_returns_sorted[:confidence_index].sum()
    return c_var


def calculate_cvar2(arguments, returns, alpha):

    w = np.array(arguments)
    portfolio_returns = np.dot(returns, w)
    mu_p = portfolio_returns.mean()
    sigma_p = portfolio_returns.std()
    c_var_p = (alpha ** -1) * norm.pdf(norm.ppf(alpha)) * sigma_p - mu_p

    return c_var_p


def calculate_cvar3(arguments, returns, alpha):

    w = np.array(arguments)
    portfolio_returns = np.dot(returns, w)
    mu_p = portfolio_returns.mean()
    sigma_p = portfolio_returns.std()
    nu = 6
    xanu = t.ppf(alpha, nu)

    c_var_p = -1 / alpha * (1 - nu) ** -1 * (nu - 2 + xanu ** 2) * t.pdf(xanu, nu) * sigma_p - mu_p

    return c_var_p


def calculate_var(arguments, returns, alpha):
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
    alpha_index = round(alpha * n_returns)
    var = portfolio_returns_sorted[alpha_index]

    return var


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


def port_return(weights_m, mu_m):
    return np.dot(np.transpose(weights_m), mu_m)


def port_var(weights_m, sigma_m):
    return np.dot(np.dot(np.transpose(weights_m), sigma_m), weights_m)


def port_sharpe(return_p, risk_p, rf):
    return (return_p - rf) / risk_p


def efficient_weights_mean_variance(cov_m, mu0, mu_m):
    matrix_a = np.zeros((len(cov_m) + 2, len(cov_m) + 2))
    matrix_a[0:len(cov_m), 0:len(cov_m)] = cov_m * 2
    matrix_a[len(cov_m), 0:len(cov_m)] = 1
    matrix_a[len(cov_m) + 1, 0:len(cov_m)] = np.transpose(mu_m)
    matrix_a[0:len(cov_m), len(cov_m)] = 1
    matrix_a[0:len(cov_m), len(cov_m) + 1] = list(mu_m)

    matrix_b = np.zeros((len(mu_m) + 2, 1))
    matrix_b[len(mu_m), 0] = 1
    matrix_b[len(mu_m) + 1, 0] = mu0

    opt = np.dot(np.linalg.inv(matrix_a), matrix_b)
    return opt[:-2]


def efficient_frontier_c_var(mu_null_start, mu_null_increment, mu_null_iterations, return_data, alpha):
    plot_data = {}
    total_portfolios = mu_null_iterations
    fail_count = 0
    cvar_large_count = 0
    for i in range(total_portfolios):
        sys.stdout.write('\r')
        sys.stdout.write('Percent Complete:  ' + str(round((i / total_portfolios) * 100, 2)) + '%')
        sys.stdout.flush()

        df_returns = return_data
        mu_null_value = mu_null_increment * i + mu_null_start
        mu_matrix = df_returns.mean()
        covariance_matrix = df_returns.cov()

        cons = [{'type': 'eq', 'fun': constraint_sum},
                {'type': 'eq', 'fun': constraint_mu_null, 'args': (mu_matrix, mu_null_value,)}]



        optimize = scipy.optimize.minimize(calculate_var,
                                           x0=np.full((len(df_returns.columns)), 1),
                                           args=(df_returns, alpha),
                                           constraints=cons,
                                           options={'maxiter': 20000})

        if not optimize.success:
            fail_count += 1
            continue

        if optimize.fun < -100:
            cvar_large_count += 1
            continue

        weights = optimize.x
        port_return = mu_null_value
        port_risk = math.sqrt(np.dot(np.dot(np.transpose(weights), covariance_matrix), weights))

        plot_data[port_risk] = port_return
    print(fail_count)
    print(cvar_large_count)
    return plot_data


def efficient_frontier_m_variance(mu_null_start, mu_null_increment, mu_null_iterations, return_data):

    plot_data = {}
    for i in range(mu_null_iterations):
        sys.stdout.write('\r')
        sys.stdout.write('Percent Complete:  ' + str(round((i / mu_null_iterations) * 100, 2)) + '%')
        sys.stdout.flush()

        mu_null_value = mu_null_increment * i + mu_null_start
        mu_matrix = np.transpose(return_data.mean())
        covariance_matrix = return_data.cov()

        eff_weights = efficient_weights_mean_variance(covariance_matrix, mu_null_value, mu_matrix)
        port_return = mu_null_value
        port_variance = port_var(eff_weights, covariance_matrix)[0][0]

        port_risk = math.sqrt(port_variance)

        plot_data[port_risk] = port_return

    return plot_data


cvar_ef = efficient_frontier_c_var(0.0001, 0.0001, 10, df_returns_hf, 0.05)
mvar_ef = efficient_frontier_m_variance(0.0001, 0.0001, 10, df_returns_hf)

plt.plot(mvar_ef.keys(), mvar_ef.values(), label='M-Variance')
plt.plot(cvar_ef.keys(), cvar_ef.values(), label='C-VaR')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.legend()
plt.show()
