import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, invwishart, invgamma
import datetime
import sys
import os
from numpy.linalg import det, inv
from codebase.plot import *
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


def ff2(yy, model_mu, model_Sigma, p):
    sample_S = np.cov(yy, rowvar=False)
    ldS = np.log(det(sample_S))
    iSigma = inv(model_Sigma)
    ldSigma = np.log(det(model_Sigma))
    n_data = yy.shape[0]
    ff2 =(n_data-1)*(ldSigma+np.sum(np.diag(sample_S @ iSigma))-ldS-p)
    return ff2


def compute_D(mcmc_iter, pred=True):
    p = ps['alpha'][mcmc_iter].shape[0]
    if pred == True:
        y_pred=multivariate_normal.rvs(mean= ps['alpha'][mcmc_iter],
                        cov=ps['Marg_cov'][mcmc_iter],
                       size = data['yy'].shape[0])
        return ff2(y_pred, ps['alpha'][mcmc_iter], ps['Marg_cov'][mcmc_iter],
            p = p)

    else:
        return ff2(data['yy'], ps['alpha'][mcmc_iter], ps['Marg_cov'][mcmc_iter],
            p = p)


mcmc_length = ps['alpha'].shape[0]
Ds = np.empty((mcmc_length,2))
for mcmc_iter in range(mcmc_length):
    Ds[mcmc_iter,0] = compute_D(mcmc_iter, pred=False)
    Ds[mcmc_iter,1] = compute_D(mcmc_iter, pred=True)


result = np.round(100*(np.sum(Ds[:,0] < Ds[:,1]) / mcmc_length),0)
print("PPP = %d %%"%result)
