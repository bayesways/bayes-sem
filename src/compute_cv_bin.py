import numpy as np
import pandas as pd
import pystan
import os
from codebase.file_utils import save_obj, load_obj
from codebase.model_fit_bin import get_lgscr
import argparse
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-nsim", "--nsim_ppp",
                    help="number of posterior samples to use for PPP", type=int, default=100)

args = parser.parse_args()

logdir = args.logdir
if logdir[-1] != "/":
    logdir = logdir + "/"

############################################################
################ Load Model Data  ##########
complete_data = load_obj("complete_data", logdir)
if 'n_splits' not in complete_data.keys():
    complete_data['n_splits'] = 3
ps = dict()
for fold_index in range(complete_data['n_splits']):
    ps[fold_index] = load_obj('ps_%s'%str(fold_index), logdir)
mcmc_length = ps[0]['alpha'].shape[0]
num_chains = ps[0]['alpha'].shape[1]
Ds_model = np.empty((complete_data['n_splits'], args.nsim_ppp, num_chains))

for fold_index in range(complete_data['n_splits']):
    Ds_model[fold_index] = get_lgscr(
        ps[fold_index],
        complete_data[fold_index],
        args.nsim_ppp
        )

############################################################
################ Compare CV scores  ##########
fold_chain_average_matrix = np.mean(Ds_model, 1)
print('\nChain/Fold Average %.2f'%np.mean(fold_chain_average_matrix))
for f in range(complete_data['n_splits']):
    chain_scores = fold_chain_average_matrix[f]
    print("\nFold %d Avg =  %.2f"%(
        f,
        np.mean(chain_scores))
        )
    print(chain_scores)
