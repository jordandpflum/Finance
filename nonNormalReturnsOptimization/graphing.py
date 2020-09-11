import pandas as pd
import matplotlib.pyplot as plt

def graph_efs():
    ef_cvar_hf = pd.read_csv('ef_cvar_hf.csv', dtype=float)
    ef_cvar_hf = ef_cvar_hf.T.reset_index().iloc[1:, ]
    ef_cvar_hf.columns = ['risk', 'return']
    ef_cvar_hf = ef_cvar_hf.apply(pd.to_numeric)

    ef_cvar_norm = pd.read_csv('ef_cvar_norm.csv')
    ef_cvar_norm = ef_cvar_norm.T.reset_index().iloc[1:, ]
    ef_cvar_norm.columns = ['risk', 'return']
    ef_cvar_norm = ef_cvar_norm.apply(pd.to_numeric)

    ef_mvar_hf = pd.read_csv('ef_mvar_hf.csv')
    ef_mvar_hf = ef_mvar_hf.T.reset_index().iloc[1:, ]
    ef_mvar_hf.columns = ['risk', 'return']
    ef_mvar_hf = ef_mvar_hf.apply(pd.to_numeric)

    ef_mvar_norm = pd.read_csv('ef_mvar_norm.csv')
    ef_mvar_norm = ef_mvar_norm.T.reset_index().iloc[1:, ]
    ef_mvar_norm.columns = ['risk', 'return']
    ef_mvar_norm = ef_mvar_norm.apply(pd.to_numeric)

    plt.plot(ef_cvar_hf['risk'], ef_cvar_hf['return'], label='C-VaR: HF INDICIES & STOCKS')
    plt.plot(ef_cvar_norm['risk'], ef_cvar_norm['return'], label='C-VaR: STOCKS')
    plt.plot(ef_mvar_hf['risk'], ef_mvar_hf['return'], label='M-Variance: HF INDICIES & STOCKS')
    plt.plot(ef_mvar_norm['risk'], ef_mvar_norm['return'], label='M-Variance: STOCKS')
    plt.xlabel('Portfolio Risk (CVaR)')
    plt.ylabel('Portfolio Return')
    plt.title('Efficient Frontiers')
    plt.legend()
    plt.show()

graph_efs()

