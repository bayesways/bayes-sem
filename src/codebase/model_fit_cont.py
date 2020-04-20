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



def Nlogpdf(yy, mean, cov):
    return multivariate_normal.logpdf(yy, mean, cov, allow_singular=True)
