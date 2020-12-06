import numpy as np
import pandas as pd
import pystan
import os
from codebase.file_utils import save_obj, load_obj
from codebase.model_fit_bin import get_lgscr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir_model", help="path to files", type=str, default=None)
parser.add_argument("logdir_benchmark", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-nsim", "--nsim_ppp",
                    help="number of posterior samples to use for PPP", type=int, default=100)

args = parser.parse_args()

logdir_model = args.logdir_model
if logdir_model[-1] != "/":
    logdir_model = logdir_model + "/"


logdir_benchmark = args.logdir_benchmark
if logdir_benchmark[-1] != "/":
    logdir_benchmark = logdir_benchmark + "/"

############################################################
################ Load Model Data  ##########
complete_data_model = load_obj("complete_data", logdir_model)
model_ps = dict()
model_ps[0] = load_obj('ps_0', logdir_model)
model_ps[1] = load_obj('ps_1', logdir_model)
model_ps[2] = load_obj('ps_2', logdir_model)
mcmc_length = model_ps[0]['alpha'].shape[0]
num_chains = model_ps[0]['alpha'].shape[1]
Ds_model = np.empty((complete_data_model['n_splits'], args.nsim_ppp, num_chains))

for fold_index in range(complete_data_model['n_splits']):
    Ds_model[fold_index] = get_lgscr(
        model_ps[fold_index],
        complete_data_model[fold_index],
        args.nsim_ppp
        )


############################################################
################ Load Benchmark Data  ##########
complete_data_benchmark = load_obj("complete_data", logdir_benchmark)

assert complete_data_benchmark['n_splits'] == complete_data_model['n_splits']

benchmark_ps = dict()
benchmark_ps[0] = load_obj('ps_0', logdir_benchmark)
benchmark_ps[1] = load_obj('ps_1', logdir_benchmark)
benchmark_ps[2] = load_obj('ps_2', logdir_benchmark)
mcmc_length = benchmark_ps[0]['alpha'].shape[0]
num_chains = benchmark_ps[0]['alpha'].shape[1]
Ds_benchmark = np.empty((complete_data_benchmark['n_splits'], args.nsim_ppp, num_chains))
for fold_index in range(complete_data_benchmark['n_splits']):
    Ds_benchmark[fold_index] = get_lgscr(
        benchmark_ps[fold_index],
        complete_data_benchmark[fold_index],
        args.nsim_ppp
        )

############################################################
################ Compare CV scores  ##########
fold_chain_average_matrix = np.mean(Ds_model<Ds_benchmark, 1)
print('\nChain/Fold Average %.2f'%np.mean(fold_chain_average_matrix))
for f in range(complete_data_model['n_splits']):
    chain_scores = fold_chain_average_matrix[f]
    print("\nFold %d Avg =  %.2f"%(
        f,
        np.mean(chain_scores))
        )
    print(chain_scores)
