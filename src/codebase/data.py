import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, bernoulli
from scipy.special import expit, logit


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
             cross_loadings = False, cross_loadings_level = 2,
             random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    if cross_loadings:
        if cross_loadings_level == 1:
            beta = np.array([[1,0.2],
                             [b, -0.3],
                             [b,-.05],
                             [-0.2,1],
                             [-.08,b],
                             [0.15,b]], dtype=float)
        elif cross_loadings_level == 2:
            beta = np.array([[1, 0],
                             [b, 0],
                             [b, .4],
                             [.4, 1],
                             [0, b],
                             [0, b]], dtype=float)
        elif cross_loadings_level == 3:
            beta = np.array([[1, 0],
                             [b, .5],
                             [b, .5],
                             [.5, 1],
                             [.5, b],
                             [0, b]], dtype=float)
        else:
            print('Noisy Level should be in [1,2,3]')

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
    data['off_diag_residual'] = off_diag_residual
    data['cross_loadings'] = cross_loadings

    return(data)


def gen_data_binary(nsim_data, J=6, K=2, rho =0.2, c=0.65, b=0.8,
             off_diag_residual = False, off_diag_corr = 0.2,
             cross_loadings = False, cross_loadings_level = 1,
             random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    if cross_loadings:
        if cross_loadings_level == 0:
            beta = np.array([[1, 0],
                             [b, 0],
                             [b,.2],
                             [0, 1],
                             [0,b],
                             [.2, b]], dtype=float)
        elif cross_loadings_level == 1:
            beta = np.array([[1, 0],
                             [b, 0],
                             [b,.5],
                             [0, 1],
                             [.5,b],
                             [0, b]], dtype=float)

        elif cross_loadings_level == 2:
            beta = np.array([[1, 0],
                             [b,.4],
                             [b,.4],
                             [0, 1],
                             [.4,b],
                             [.4, b]], dtype=float)
        else:
            print('Noisy Level should be in [0,1,2]')
    else:
        beta = np.array([[1,0],
                         [b,0],
                         [b,0],
                         [0,1],
                         [0,b],
                         [0,b]], dtype=float)

    alpha = np.zeros(J)

    sigma_z = np.repeat(np.sqrt(c), K)
    Phi_corr = np.eye(K)
    Phi_corr[0,1] = rho
    Phi_corr[1,0] = rho
    Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)

    Theta = np.eye(J)
    if off_diag_residual:
        for i in [1,2,5]:
            for j in [3,4]:
                Theta[i,j] = off_diag_corr
                Theta[j,i] = off_diag_corr

    zz = multivariate_normal.rvs(mean = np.zeros(K), cov=Phi_cov, size=nsim_data)
    yy = alpha + zz @ beta.T

    # ee = None
    # logit method
    # DD = bernoulli.rvs(p=expit(yy))

    # probit method
    # DD = bernoulli.rvs(p=norm.cdf(yy))

    ee_seed = multivariate_normal.rvs(mean = np.zeros(J), cov=Theta, size=nsim_data)
    ee = logit(norm.cdf(ee_seed))
    yy = alpha + zz @ beta.T + ee
    DD = (yy>0).astype(int)


    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['Theta'] = Theta
    data['e'] = ee
    data['sigma_z'] = sigma_z
    data['Phi_corr'] = Phi_corr
    data['Phi_cov'] = Phi_cov
    data['y'] = yy
    data['D'] = DD
    data['off_diag_residual'] = off_diag_residual
    data['cross_loadings'] = cross_loadings

    return(data)


def gen_data_binary_1factor(nsim_data, J=6, K=1, c=1, noise=False,
        cheaters = False, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    # alpha = np.array([-0.53,  0.35, -1.4 , -1.4 , -0.96, -2.33])
    alpha = np.zeros(J)
    beta = np.array([1, 0.7, .8  , .5, .9, .6])


    zz = norm.rvs(scale = c, size=nsim_data)
    yy = alpha + np.outer(zz, beta)
    if cheaters: # add cheaters
        yy[900:,3:]= yy[900:,3:] + 0.5

    DD = bernoulli.rvs(p=expit(yy))

    if noise: # replace noisy column
        noisy_col = bernoulli.rvs(p=0.5, size=nsim_data)
        DD[:,0] = noisy_col

    data = dict()
    data['noise'] = noise
    data['cheaters'] = cheaters
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['sigma'] = c
    data['z'] = zz
    data['y'] = yy
    data['D'] = DD

    return(data)
