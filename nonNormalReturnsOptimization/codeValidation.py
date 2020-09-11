import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from NonNormalReturnOptimization.Optimization import efficient_frontier_cvar, efficient_frontier_m_variance


mu, sigma = .1, 0.4 # mean and standard deviation
a = np.random.normal(mu, sigma, 1000)
a = sorted(a)

mu, sigma = .1, 0.4 # mean and standard deviation
b = np.random.normal(mu, sigma, 1000)
b = sorted(b)

mu, sigma = .1, 0.4 # mean and standard deviation
c = np.random.normal(mu, sigma, 1000)
c = sorted(c)

x = pd.DataFrame({'a': a, 'b': b, 'c': c})

n_ef_cvar = efficient_frontier_cvar(-0.25, 0.1, 30, x, 95)
n_ef_mvar = efficient_frontier_m_variance(-0.25, 0.1, 30, x)


plt.plot(n_ef_mvar.keys(), n_ef_mvar.values(), label='M-Variance')
plt.plot(n_ef_cvar.keys(), n_ef_cvar.values(), label='C-VaR')
plt.xlabel('Portfolio Risk (CVaR)')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontiers: Using approximately normal stock returns')
plt.legend()
plt.show()

# Result should be approximately the same