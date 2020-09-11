import scipy.optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
from NonNormalReturnOptimization.portfolioUtilities import efficient_weights_mean_variance, port_variance, port_return


pd.set_option('display.max_columns', None)
pd.options.display.width = 0

df_returns_data = pd.read_csv('Data.csv', index_col=0)
df_returns_hf = pd.read_csv('Data_HF - Copy.csv', index_col=0).dropna()


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
        port_var = port_variance(eff_weights, covariance_matrix)[0][0]
        port_risk = calculate_cvar(eff_weights, return_data, 95)

        plot_data[port_risk] = port_return

    return plot_data


def efficient_frontier_cvar(mu_null_start, mu_null_increment, mu_null_iterations, return_data, confidence):

    plot_data = {}
    for i in range(mu_null_iterations):
        sys.stdout.write('\r')
        sys.stdout.write('Percent Complete:  ' + str(round((i / mu_null_iterations) * 100, 2)) + '%')
        sys.stdout.flush()

        mu_null_value = mu_null_increment * i + mu_null_start
        mu_matrix = np.transpose(return_data.mean())
        covariance_matrix = return_data.cov()

        eff_weights = efficient_weights_cvar(return_data, mu_null_value, confidence)
        port_return = mu_null_value
        port_var = port_variance(eff_weights, covariance_matrix)
        port_risk = calculate_cvar(eff_weights, return_data, 95)

        plot_data[port_risk] = port_return

    return plot_data


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


def calculate_cvar(weights, returns, confidence):
    """
    Calculates the conditional value at risk at a given confidence level
    :param weights: the iterative weights thrown in by the optimizer
    :param returns: data frame of returns for all securities
    :param confidence: confidence level for value at risk
    :return: the conditional value at risk at a given confidence level at the portfolio level
    """

    w = np.array(weights)
    portfolio_returns = np.dot(returns, w)  # calculate the time series of portfolio returns given weights

    portfolio_returns_sorted = np.sort(portfolio_returns, axis=0)  # sort these returns in ascending order
    n_returns = len(portfolio_returns_sorted)  # get the number of returns in this time period
    confidence_index = round(((100 - confidence) * 0.01) * n_returns)  # find the return position in the list of sorted
    #  returns given a confidence level. This is considered the Value at Risk

    c_var = (1 / confidence_index) * portfolio_returns_sorted[:confidence_index].sum()  # get the average return
    #  below the VaR. These are still in negative form and sorted as ascending, so we want the sum of the lowest return
    #  to the VaR return.

    return -c_var  # we negate this value because we are trying to minimize a loss. Loss functions are positive.


def efficient_weights_cvar(returns, mu0, confidence):
    mu_matrix = returns.mean()

    cons = [{'type': 'eq', 'fun': constraint_sum},
            {'type': 'eq', 'fun': constraint_mu_null, 'args': (mu_matrix, mu0)}]

    minimizer_kwargs = {"args": (returns, confidence),
                        "constraints": cons}

    optimize = scipy.optimize.basinhopping(calculate_cvar,
                                           x0=np.full((len(returns.columns)), 1 / len(returns.columns)),
                                           minimizer_kwargs=minimizer_kwargs)
    return optimize.x


def plot_both_efficient_frontiers(data):

    cvar_ef = efficient_frontier_cvar(0.0001, 0.0001, 50, data, 95)
    mvar_ef = efficient_frontier_m_variance(0.0001, 0.0001, 50, data)

    plt.plot(mvar_ef.keys(), mvar_ef.values(), label='M-Variance')
    plt.plot(cvar_ef.keys(), cvar_ef.values(), label='C-VaR')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.legend()
    plt.show()


def maximize_cvar_sharpe(weights, return_data, risk_free, confidence):
    return_p = port_return(weights, return_data.mean())
    cvar = calculate_cvar(weights, return_data, confidence)

    return -(return_p - risk_free) / cvar


def maximize_variance_sharpe(weights, return_data, risk_free):
    return_p = port_return(weights, return_data.mean())
    standard_deviation_p = math.sqrt(port_variance(weights, return_data.cov()))

    return -(return_p - risk_free) / standard_deviation_p


def optimal_portfolio_weights_cvar(returns_data, confidence, risk_free):

    cons = [{'type': 'eq', 'fun': constraint_sum}]

    minimizer_kwargs = {"args": (returns_data, risk_free, confidence),
                        "constraints": cons}

    optimize = scipy.optimize.basinhopping(maximize_cvar_sharpe,
                                           x0=np.full((len(returns_data.columns)), 1 / len(returns_data.columns)),
                                           minimizer_kwargs=minimizer_kwargs)

    return optimize.x


def optimal_portfolio_weights_variance(returns_data, risk_free):
    cons = [{'type': 'eq', 'fun': constraint_sum}]

    minimizer_kwargs = {"args": (returns_data, risk_free),
                        "constraints": cons}

    optimize = scipy.optimize.basinhopping(maximize_variance_sharpe,
                                           x0=np.full((len(returns_data.columns)), 1 / len(returns_data.columns)),
                                           minimizer_kwargs=minimizer_kwargs)

    return optimize.x


def plot_everything(data, direc):
    cvar_ef = efficient_frontier_cvar(0.0001, 0.0001, 1000, data, 95)
    mvar_ef = efficient_frontier_m_variance(0.0001, 0.0001, 1000, data)
    opt = 'cvar'
    url = f'ef_{opt}_{direc}.csv'
    pd.DataFrame(cvar_ef, index=[0]).to_csv(url)
    opt = 'mvar'
    url = f'ef_{opt}_{direc}.csv'
    pd.DataFrame(mvar_ef, index=[0]).to_csv(url)
