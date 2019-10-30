import numpy as np
import pandas as pd
import pystan
from scipy.stats import multivariate_normal, bernoulli
from scipy.special import expit


def flatten_df(df0, val_name, var_name = 'K'):
    df = pd.DataFrame(df0)
    df.columns  = df.columns + 1
    df['J'] = np.arange(len(df))+1
    return df.melt(id_vars=['J'], var_name=var_name, value_name = val_name)


def post_summary(samples):
    mean =  pd.DataFrame(np.mean(samples, axis=0))
    ps_df = flatten_df(mean, 'mean')
    median = pd.DataFrame(np.median(samples, axis=0))
    ps_df['median'] = flatten_df(median, 'median')['median']
    per1 = pd.DataFrame(np.percentile(samples, 2.5,axis=0))
    ps_df['q2.5'] = flatten_df(per1, 'q2.5')['q2.5']
    per2 = pd.DataFrame(np.percentile(samples, 97.5,axis=0))
    ps_df['q97.5'] = flatten_df(per2, 'q97.5')['q97.5']
    return ps_df


def C_to_R(M):
    """
    Send a covariance matrix M to the corresponding
    correlation matrix R
    Inputs
    ============
    - M : covariance matrix
    Output
    ============
    - correlation matrix
    """
    d = np.asarray(M.diagonal())
    d2 = np.diag(d**(-.5))
    R = d2 @ M @ d2
    return R


def gen_data(nsim_data, J=6, K=2, rho =0.2, c=0.65, b=0.8,
             off_diag_residual = False, off_diag_corr = 0.2,
             noisy_loadings = False, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    if noisy_loadings:
        # beta = np.array([[1,0.2],
        #                  [b, -0.3],
        #                  [b,-.05],
        #                  [-0.2,1],
        #                  [-.08,b],
        #                  [0.15,b]], dtype=float)
        beta = np.array([[1, 0],
                         [b, 0],
                         [b, .4],
                         [0, 1],
                         [.4, b],
                         [0, b]], dtype=float)

    else:
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
    data['random_seed'] = random_seed
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


def gen_data_binary(nsim_data, J=6, K=2, rho =0.2, c=0.65, b=0.8,
             off_diag_residual = False, off_diag_corr = 0.2,
             noisy_loadings = False, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    if noisy_loadings:
        # beta = np.array([[1,0.2],
        #                  [b, -0.3],
        #                  [b,-.05],
        #                  [-0.2,1],
        #                  [-.08,b],
        #                  [0.15,b]], dtype=float)
        beta = np.array([[1, 0],
                         [b, .5],
                         [b, .5],
                         [.5, 1],
                         [.5, b],
                         [0, b]], dtype=float)

    else:
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
    DD = bernoulli.rvs(p=expit(yy))

    data = dict()
    data['random_seed'] = random_seed
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
    data['D'] = DD

    return(data)
