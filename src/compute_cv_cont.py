import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
from codebase.model_fit_cont import get_lgscr


from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-nsim", "--nsim_ppp",
                    help="number of posterior samples to use for PPP", type=int, default=100)


args = parser.parse_args()

print("\n\nPrinting Stan model code \n\n")


logdir = args.logdir
if logdir[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    logdir = logdir + "/"

############################################################
################ Load Data and models  ##########

print("\n\nLoading files...\n\n")

complete_data = load_obj("complete_data", logdir)


model_posterior_samples = dict()

model_posterior_samples = dict()
model_posterior_samples[0] = load_obj('ps_0', logdir)
model_posterior_samples[1] = load_obj('ps_1', logdir)
model_posterior_samples[2] = load_obj('ps_2', logdir)

print("\n\nComputing Folds...\n\n")

mcmc_length = model_posterior_samples[0]['alpha'].shape[0]
num_chains = model_posterior_samples[0]['alpha'].shape[1]


Ds = np.empty((3, num_chains))
for fold_index in range(3):
    lgscr_vals = get_lgscr(model_posterior_samples[fold_index], complete_data[fold_index],
                           args.nsim_ppp)

    # for each chain take the mean log_score across the MCMC iters
    Ds[fold_index] = np.mean(lgscr_vals, 0)

# for each chain, sum the log_scores across 3 folds
logscore_chains = np.sum(Ds, axis=0)
avg_logscore =  np.round(np.mean(logscore_chains),4) # take the mean sum log-score)

avg_logscore_folds =  np.round(np.mean(Ds, axis=1),2) # take the mean sum log-score)
print(avg_logscore_folds)

print("Log score for each chain", logscore_chains)
print("Avg Log score", np.int(avg_logscore))
