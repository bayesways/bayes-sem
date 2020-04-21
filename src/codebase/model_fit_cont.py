import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from numpy.linalg import det, inv
from codebase.file_utils import save_obj, load_obj
from scipy.special import expit


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


def Nlogpdf(yy, mean, cov):
    return multivariate_normal.logpdf(yy, mean, cov, allow_singular=True)


def get_lgscr(ps, data, nsim):
    mcmc_length = ps['alpha'].shape[0]
    num_chains = ps['alpha'].shape[1]
    skip_step = int(mcmc_length/nsim)

    lgscr_vals = np.empty((nsim, num_chains))
    for m_ind in range(nsim):
            m = m_ind * skip_step
            for cn in range(num_chains):
                model_lgpdf = Nlogpdf(data['test']['yy'],
                                    ps['alpha'][m, cn],
                                    ps['Marg_cov'][m, cn])
                lgscr_vals[m_ind, cn] = -2*np.sum(model_lgpdf)

    return lgscr_vals