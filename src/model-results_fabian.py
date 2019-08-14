import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, invwishart, invgamma
import datetime
import sys
import os
from numpy.linalg import det, inv
from codebase.file_utils import save_obj, load_obj


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=1)

args = parser.parse_args()

print("\n\nPrinting Stan model code \n\n")


log_dir = args.logdir
if log_dir[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    log_dir = log_dir+ "/"

if bool(args.print_model):
    file = open(log_dir+"model.txt", "r")
    print(file.read())
    file.close()


data = load_obj("stan_data", log_dir)
ps = load_obj('ps', log_dir)

def ff (yy, model_mu, model_Sigma, p=15, q=5):
    mle_est = dict()
    sample_S = np.cov(yy, rowvar=False)
    sample_m = np.mean(yy, axis=0)
    n_data = yy.shape[0]

    term1 = np.log(det(model_Sigma))
    term2 = inv(model_Sigma) @ (sample_S + (sample_m - model_mu) @  (sample_m - model_mu))
    term3 = np.log(det(sample_S)) + p + q

    ff = 0.5 * n_data * ( term1 + np.trace(term2)) - term3

    return ff


if 'Marg_cov2' in ps.keys():
    marg_cov = 'Marg_cov2'
else:
    marg_cov = 'Marg_cov'


def compute_D(mcmc_iter, pred=True):
    if pred == True:
        y_pred=multivariate_normal.rvs(mean= ps['alpha'][mcmc_iter],
                        cov=ps[marg_cov][mcmc_iter],
                       size = data['yy'].shape[0])
        return ff(y_pred, ps['alpha'][mcmc_iter], ps[marg_cov][mcmc_iter])

    else:
        return ff(data['yy'], ps['alpha'][mcmc_iter], ps[marg_cov][mcmc_iter])


mcmc_length = ps['alpha'].shape[0]
Ds = np.empty((mcmc_length,2))
for mcmc_iter in range(mcmc_length):
    Ds[mcmc_iter,0] = compute_D(mcmc_iter, pred=False)
    Ds[mcmc_iter,1] = compute_D(mcmc_iter, pred=True)


print(np.sum(Ds[:,0] < Ds[:,1]) / mcmc_length)
