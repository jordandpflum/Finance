import scipy.optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
from scipy.stats import norm
from scipy.stats import t
import random


pd.set_option('display.max_columns', None)
pd.options.display.width = 0

df_returns_data = pd.read_csv('C:\\Users\\jason\\PycharmProjects\\UT MSBA\\Decision Analytics\\Data.csv', index_col=0)
df_returns_hf = pd.read_csv('Data_HF - Copy.csv', index_col=0).dropna()[['HFRX Global Hedge Fund Index (HFRXGL)', 'HFRX Absolute Return Index (HFRXAR)', 'HFRX Aggregate Index (HFRXAGGR)']]


# print(df_returns_data)

def port_return(weights_m, mu_m):
    return np.dot(np.transpose(weights_m), mu_m)

def calculate_mu(return_data):
    """
    Calculates the Mu matrix for the securities
    :param return_data: the data frame containing the returns
    :return: returns an array containing the arithmetic average return
    """
    return np.array(return_data.mean())


def var_distribution(weights, returns, confidence, num_calcs, period):
    """
    Calculates the conditional value at risk at a given confidence level
    :param arguments: the iterative weights thrown in by the optimizer
    :param returns: data frame of returns for all securities
    :param confidence: confidence level for value at risk
    :return: the conditional value at risk at a given confidence level at the portfolio level
    """

    # w = np.array(weights)
    portfolio_returns = np.dot(returns, weights)

    return_distribution = []
    for i in range(num_calcs):
        period_return_distribution = 0
        sys.stdout.write('\r')
        sys.stdout.write('Percent Complete:  ' + str(round((i / num_calcs) * 100, 2)) + '%')
        sys.stdout.flush()

        for j in range(period):
            random_index = int(random.random() * len(df_returns_data))
            instance_return = portfolio_returns[random_index]
            period_return_distribution += instance_return

        return_distribution.append(period_return_distribution / period)

    # print(return_distribution)
    print(port_return(weights, calculate_mu(returns)))
    print(np.quantile(return_distribution, confidence))

def calculate_var(weights, returns, confidence, num_calcs, period):
    """
    Calculates the conditional value at risk at a given confidence level
    :param arguments: the iterative weights thrown in by the optimizer
    :param returns: data frame of returns for all securities
    :param confidence: confidence level for value at risk
    :return: the conditional value at risk at a given confidence level at the portfolio level
    """

    portfolio_returns = np.dot(returns, weights)

    return_distribution = []
    for i in range(num_calcs):
        period_return_distribution = 0
        # sys.stdout.write('\r')
        # sys.stdout.write('Percent Complete:  ' + str(round((i / num_calcs) * 100, 2)) + '%')
        # sys.stdout.flush()

        for j in range(period):
            random_index = int(random.random() * len(df_returns_data))
            instance_return = portfolio_returns[random_index]
            period_return_distribution += instance_return

        return_distribution.append(period_return_distribution / period)

    return np.quantile(return_distribution, confidence)

# weights = np.full((len(df_returns_data.columns), 1), 1 / len(df_returns_data))
# weights = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,-13999]
# var_distribution(weights, df_returns_data, .01, 5000, 10)
# weights = [10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,10000,-139999]
# var_distribution(weights, df_returns_data, .01, 5000, 10)
# weights = [100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,-1399999]
# var_distribution(weights, df_returns_data, .01, 5000, 10)


def port_var(weights_m, sigma_m):
    return np.dot(np.dot(np.transpose(weights_m), sigma_m), weights_m)

def efficient_weights_mean_variance(cov_m, mu0, mu_m):
    # print(cov_m)
    # print(mu_m)
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



def efficient_frontier_m_variance_optimizer(mu_null_start, mu_null_increment, mu_null_iterations, return_data):

    plot_data = {}
    for i in range(mu_null_iterations):
        sys.stdout.write('\r')
        sys.stdout.write('Percent Complete:  ' + str(round((i / mu_null_iterations) * 100, 2)) + '%')
        sys.stdout.flush()

        mu_null_value = mu_null_increment * i + mu_null_start
        mu_matrix = np.transpose(return_data.mean())
        covariance_matrix = return_data.cov()

        cons = [{'type': 'eq', 'fun': constraint_sum},
                {'type': 'eq', 'fun': constraint_mu_null, 'args': (mu_matrix, mu_null_value)}]

        optimize = scipy.optimize.minimize(port_var,
                                           x0=np.full((len(return_data.columns)), 1 / len(return_data.columns)),
                                           args=(covariance_matrix),
                                           constraints=cons)



        port_return = mu_null_value
        port_variance = port_var(optimize.x, covariance_matrix)
        print(port_variance == optimize.fun)

        port_risk = math.sqrt(port_variance)

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

def efficient_frontier_var(mu_null_start, mu_null_increment, mu_null_iterations, return_data, alpha):
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
                                           args=(df_returns, alpha, 10, 10),
                                           constraints=cons)

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


# cvar_ef = efficient_frontier_var(0.0001, 0.0001, 10, df_returns_hf, 0.05)
# mvar_ef = efficient_frontier_m_variance(0.0001, 0.0001, 1000, df_returns_data)
# mvar_ef2 = efficient_frontier_m_variance_optimizer(0.0001, 0.0001, 1000, df_returns_data)
#
# print(mvar_ef)
# print(mvar_ef2)
#
# plt.plot(mvar_ef.keys(), mvar_ef.values(), label='M-Variance')
# plt.plot(mvar_ef2.keys(), mvar_ef2.values(), label='M-Variance2')
# plt.plot(cvar_ef.keys(), cvar_ef.values(), label='C-VaR')
# plt.xlabel('Risk')
# plt.ylabel('Return')
# plt.legend()
# plt.show()

def mean_variance_optimizer(return_data, mu_null):

    mu_matrix = np.transpose(return_data.mean())
    # print(mu_matrix)
    covariance_matrix = return_data.cov()
    # print(covariance_matrix)
    cons = [{'type': 'eq', 'fun': constraint_sum},
            {'type': 'eq', 'fun': constraint_mu_null, 'args': (mu_matrix, mu_null)}]

    # optimize = scipy.optimize.minimize(port_var,
    #                                    x0=np.full((len(return_data.columns)), 1 / len(return_data.columns)),
    #                                    args=(covariance_matrix),
    #                                    constraints=cons,
    #                                    method='SLSQP')

    minimizer_kwargs = {"args": covariance_matrix,
                        "constraints": cons}

    optimize = scipy.optimize.basinhopping(port_var,
                                           x0=np.full((len(return_data.columns)), 1),
                                           minimizer_kwargs=minimizer_kwargs)


    port_variance = port_var(optimize.x, covariance_matrix)
    port_risk = math.sqrt(port_variance)


    return port_risk

print(mean_variance_optimizer(df_returns_data, 0.05))
print(math.sqrt(port_var(efficient_weights_mean_variance(df_returns_data.cov(), 0.05, df_returns_data.mean()), df_returns_data.cov())))
