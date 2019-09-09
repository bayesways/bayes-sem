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
    Phi_corr = np.eye(K)
    Phi_corr[0,1] = rho
    Phi_corr[1,0] = rho
    Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)

    sigma = np.array([1,1.2,.9,.8, 1, 1.4])
    Theta = np.diag(sigma**2)

    Marg_cov = beta @ Phi_cov @ beta.T + Theta
    yy = multivariate_normal.rvs(mean = alpha, cov=Marg_cov, size=nsim_data)



    data = dict()
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['sigma_z'] = sigma_z
    data['Phi_corr'] = Phi_corr
    data['Phi_cov'] = Phi_cov
    data['Marg_cov'] = Marg_cov
    data['Theta'] = Theta
    data['sigma'] = sigma
    data['y'] = yy

    return(data)
