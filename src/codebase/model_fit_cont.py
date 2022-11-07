import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from numpy.linalg import det, inv, norm
from codebase.file_utils import save_obj, load_obj
from scipy.special import expit
from scipy.spatial.distance import pdist
from pdb import set_trace


def ff2(yy, model_mu, model_Sigma, p):
    sample_S = np.cov(yy, rowvar=False)
    ldS = np.log(det(sample_S))
    iSigma = inv(model_Sigma)
    ldSigma = np.log(det(model_Sigma))
    n_data = yy.shape[0]
    ff2 = (n_data-1)*(ldSigma+np.sum(np.diag(sample_S @ iSigma))-ldS-p)
    return ff2


def compute_D(data, ps, mcmc_iter, pred=True):
    J = ps['alpha'][mcmc_iter].shape[0]
    if pred == True:
        y_pred = multivariate_normal.rvs(mean=ps['alpha'][mcmc_iter],
                                         cov=ps['Marg_cov'][mcmc_iter],
                                         size=data['yy'].shape[0])
        return ff2(y_pred, ps['alpha'][mcmc_iter], ps['Marg_cov'][mcmc_iter],
                   p=J)

    else:
        return ff2(data['yy'], ps['alpha'][mcmc_iter], ps['Marg_cov'][mcmc_iter],
                   p=J)

def Nlogpdf(yy, mean, cov):
    return multivariate_normal.logpdf(yy, mean, cov, allow_singular=True)

def get_PPP(data, ps, nsim):
    mcmc_length = ps['alpha'].shape[0]
    skip_step = int(mcmc_length/nsim)
    PPP_vals = np.empty((nsim, 2))
    for m_ind in tqdm(range(nsim)):
        m = skip_step*m_ind
        PPP_vals[m_ind, 0] = compute_D(data, ps, m, pred=False)
        PPP_vals[m_ind, 1] = compute_D(data, ps, m, pred=True)
    return PPP_vals


def energy_score(y_pred, y_obs):
    # y_obs has dim J (1 observation)
    # y_pred has dim m x J (m posterior samnples)
    m = y_pred.shape[0]
    # sum ||Xi - y|| for all i
    s1 = norm((y_pred - y_obs), ord=2, axis=1).sum() 
    # sum all distinct pairs ||Xi - Xj|| for all i and all j<i
    # note: this results in summing Xi - Xj only once 
    s2 = pdist(y_pred, metric='euclidean').sum()
    
    # remove the 1/2 from the second sum because we summed each distinct pair only once
    return ((1./m)*s1) - ((1./m**2)*s2)

def variogram_score(y_pred, y_obs, p = 0.5):
    mcmc_lenght = y_pred.shape[0]
    y_obs_length = y_obs.shape[0]
    p = 0.5

    s = 0.
    for i in range(y_obs_length):
        for j in range(y_obs_length):
            s1 = np.abs(y_obs[i] - y_obs[j])**p
            s2 = (-1./mcmc_lenght) * np.array(
                [
                    np.abs(y_pred[k,i] - y_pred[k,j])**p
                    for k in range(mcmc_lenght)
                ]).sum()
            s_ij = (s1 + s2)**2
            s = s + s_ij
    return s

def energy_score_vector(y_pred, y_obs_vector):
    # y_obs has dim N x J (N observation)
    # y_pred has dim m x J (m posterior samnples)
    N = y_obs_vector.shape[0]
    scores = np.empty(N)
    for i in tqdm(range(N)):
        scores[i] = energy_score(y_pred, y_obs_vector[i])
    return scores


def variogram_score_vector(y_pred, y_obs_vector):
    # y_obs has dim N x J (N observation)
    # y_pred has dim m x J (m posterior samnples)
    N = y_obs_vector.shape[0]
    scores = np.empty(N)
    for i in tqdm(range(N)):
        scores[i] = variogram_score(y_pred, y_obs_vector[i])
    return scores


def get_log_score(ps, data, nsim=10, method_num=2):
    mcmc_length = ps["alpha"].shape[0]
    dim_J = ps['alpha'].shape[1]
    if nsim>mcmc_length:
        print('nsim > posterior sample size')
        print('Using nsim = %d'%mcmc_length)
        nsim = mcmc_length
    skip_step = int(mcmc_length/nsim)
    test_size = data.shape[0]
    theta_draws = np.empty(nsim, dtype = float)
    y_lklhds = np.empty(test_size, dtype = float)

    for i in range(test_size):
        for m_ind in range(nsim):
            m = m_ind * skip_step
            mean =  ps['alpha'][m]
            Cov = ps['Marg_cov'][m]
            theta_draws[m_ind] = multivariate_normal.pdf(
                data[i],
                mean=mean,
                cov = Cov
                )
        y_lklhds[i] = -np.log(np.mean(theta_draws))
    scores = dict()
    logscore = y_lklhds.sum()
    scores['logscore'] = logscore

    return scores    
    

def get_energy_scores(ps, data, nsim=10, method_num=2):
    mcmc_length = ps["alpha"].shape[0]
    dim_J = ps['alpha'].shape[1]
    if nsim>mcmc_length:
        print('nsim > posterior sample size')
        print('Using nsim = %d'%mcmc_length)
        nsim = mcmc_length
    skip_step = int(mcmc_length/nsim)
    post_y = np.empty((nsim, dim_J), dtype = float)
    
    # method 1 
    # use posterior mean
    if method_num == 1:
        m_alpha = ps['alpha'].mean(axis=0)
        m_Cov = ps['Marg_cov'].mean(axis=0)
        post_y = multivariate_normal.rvs(mean=m_alpha, cov = m_Cov, size=10000)  
    
    # method 2
    # draw one y sample per posterior sample theta
    elif method_num == 2:
        for m_ind in range(nsim):
            m = m_ind * skip_step
            mean =  ps['alpha'][m]
            Cov = ps['Marg_cov'][m]
            post_y[m_ind] = multivariate_normal.rvs(mean=mean, cov = Cov)
    
    # method 3 
    # use posterior median and log
    elif method_num ==3:
        m_alpha = np.median(ps['alpha'], axis=0)
        m_Cov = np.median(ps['Marg_cov'], axis=0)
        scores = -1.* Nlogpdf(data, m_alpha, m_Cov).sum()
    else:
        print("method_num not found")
    scores = dict()
    scores_variogram = variogram_score_vector(
        y_pred=post_y,
        y_obs_vector=data).sum()
    scores['variogram'] = scores_variogram

    return scores