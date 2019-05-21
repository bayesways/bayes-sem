import numpy as np
import pandas as pd
import pystan
from scipy.stats import multivariate_normal

def gen_data(nsim_data, J, K, rho =0.5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.array([1,2,.3,-.8, 1, -1.4])
    beta = np.array([[1,0], [.2, 0],[.6,0],[0,1], [0,.5], [0,.8]], dtype=float)
    sigma_z = np.array([1.2, .7])
    V_corr = np.eye(K)
    V_corr[0,1] = rho
    V_corr[1,0] = rho
    V = np.diag(sigma_z) @ V_corr @  np.diag(sigma_z)

    sigma_e = np.array([1,1.2,.9,.8, 1, 1.4])
    Sigma_e = np.diag(sigma_e**2)

    Omega = beta @ V @ beta.T + Sigma_e
    yy = multivariate_normal.rvs(mean = alpha, cov=Omega, size=nsim_data)



    data = dict()
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['sigma_z'] = sigma_z
    data['V_corr'] = V_corr
    data['V'] = V
    data['Omega'] = Omega
    data['Sigma_e'] = Sigma_e
    data['sigma_e'] = sigma_e
    data['y'] = yy

    return(data)
