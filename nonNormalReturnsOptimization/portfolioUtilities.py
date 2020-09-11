import numpy as np

def port_return(weights_m, mu_m):
    return np.dot(np.transpose(weights_m), mu_m)


def calculate_mu(return_data):
    """
    Calculates the Mu matrix for the securities
    :param return_data: the data frame containing the returns
    :return: returns an array containing the arithmetic average return
    """
    return np.array(return_data.mean())


def port_variance(weights_m, sigma_m):
    return np.dot(np.dot(np.transpose(weights_m), sigma_m), weights_m)


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