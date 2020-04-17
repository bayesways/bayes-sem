import numpy as np
import pandas as pd
import pystan
import os
from codebase.file_utils import save_obj, load_obj
from codebase.model_fit_bin import get_lgscr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-nsim", "--nsim_ppp",
                    help="number of posterior samples to use for PPP", type=int, default=100)

args = parser.parse_args()

log_dir = args.logdir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"


print("\n\nChecking data integrity...\n\n")
complete_data = load_obj("complete_data", log_dir)

model_posterior_samples = dict()
model_posterior_samples[0] = load_obj('ps_0', log_dir)
model_posterior_samples[1] = load_obj('ps_1', log_dir)
model_posterior_samples[2] = load_obj('ps_2', log_dir)


num_chains = model_posterior_samples[0]['alpha'].shape[1]
num_samples = model_posterior_samples[0]['alpha'].shape[0]

Ds = np.empty((3,num_chains))
for fold_index in range(3):
    lgscr_vals, Dy = get_lgscr(complete_data[fold_index]['test'], model_posterior_samples[fold_index], num_chains, args.nsim_ppp)
    Ds[fold_index] = np.mean(lgscr_vals,0) #for each chain take the mean log_score across the MCMC iters

logscore_chains = np.sum(Ds, axis=0) # for each chain, sum the log_scores across 3 folds

avg_logscore =  np.round(np.mean(np.sum(Ds, axis=0)),4) # take the mean sum log-score)

print("Log score for each chain", logscore_chains)
print("Avg Log score", avg_logscore)