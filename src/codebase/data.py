import numpy as np
import pandas as pd
import pystan
from scipy.stats import multivariate_normal

def gen_data(nsim_data, J=6, K=2, rho =0.2, c=0.65, b=0.8,
             off_diag_residual = False, off_diag_corr = 0.2,
             random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    b = 0.8
    beta = np.array([[1,0],
                     [b, 0],
                     [b,0],
                     [0,1],
                     [0,b],
                     [0,b]], dtype=float)

    sigma_z = np.repeat(np.sqrt(c), K)
    Phi_corr = np.eye(K)
    Phi_corr[0,1] = rho
    Phi_corr[1,0] = rho
    Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)

    sigma_sq = 1 - np.diag(beta @ Phi_cov @ beta.T)
    sigma = np.sqrt(sigma_sq)

    if off_diag_residual:
        Theta_corr = np.eye(J)
#         Theta = np.diag(sigma_sq)
        for i in [1,2,5]:
            for j in [3,4]:
#                 Theta[i,j] = off_diag_corr*sigma[i]*sigma[j]
#                 Theta[j,i] = off_diag_corr*sigma[i]*sigma[j]
                Theta_corr[i,j] = off_diag_corr
                Theta_corr[j,i] = off_diag_corr
        Theta = np.diag(np.sqrt(sigma_sq)) @ Theta_corr @  np.diag(np.sqrt(sigma_sq))
    else:
        Theta = np.diag(sigma_sq)

    Marg_cov = beta @ Phi_cov @ beta.T + Theta
    Marg_cov
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
