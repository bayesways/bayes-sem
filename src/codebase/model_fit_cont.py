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


def compute_D(data, ps, mcmc_iter, cn, pred=True):
    J = ps['alpha'][mcmc_iter, cn].shape[0]
    if pred == True:
        y_pred = multivariate_normal.rvs(mean=ps['alpha'][mcmc_iter, cn],
                                         cov=ps['Marg_cov'][mcmc_iter, cn],
                                         size=data['yy'].shape[0])
        return ff2(y_pred, ps['alpha'][mcmc_iter, cn], ps['Marg_cov'][mcmc_iter, cn],
                   p=J)

    else:
        return ff2(data['yy'], ps['alpha'][mcmc_iter, cn], ps['Marg_cov'][mcmc_iter, cn],
                   p=J)


def get_PPP(data, ps, cn, nsim):
    mcmc_length = ps['alpha'].shape[0]
    skip_step = int(mcmc_length/nsim)
    PPP_vals = np.empty((nsim, 2))
    for m_ind in tqdm(range(nsim)):
        m = skip_step*m_ind
        PPP_vals[m_ind, 0] = compute_D(data, ps, m, cn, pred=False)
        PPP_vals[m_ind, 1] = compute_D(data, ps, m, cn, pred=True)
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

def energy_score_vector(y_pred, y_obs_vector):
    # y_obs has dim N x J (N observation)
    # y_pred has dim m x J (m posterior samnples)
    N = y_obs_vector.shape[0]
    scores = np.empty(N)
    for i in range(N):
        scores[i] = energy_score(y_pred, y_obs_vector[i])
    return scores

def get_energy_scores(ps, data, nsim, cn):
    mcmc_length = ps['alpha'].shape[0]
    dim_J = ps['alpha'].shape[2]
    skip_step = int(mcmc_length/nsim)
    post_y = np.empty((nsim, dim_J), dtype = float)

    for m_ind in range(nsim):
        m = m_ind * skip_step
        mean = ps['alpha'][m, cn]
        Cov = ps['Marg_cov'][m, cn]
        post_y[m_ind] = multivariate_normal.rvs(mean=mean, cov = Cov)

    scores = energy_score_vector(
        y_pred=post_y,
        y_obs_vector=data).sum()
    return scores