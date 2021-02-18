import numpy as np
import pandas as pd
import pystan
import os
from codebase.file_utils import save_obj, load_obj
from codebase.model_fit_bin import get_scores
import argparse
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument(
    "-nsim",
    "--nsim_ppp",
    help="number of posterior samples to use for PPP",
    type=int,
    default=None,
)
parser.add_argument(
    "-scr",
    "--scr_metric",
    help="score metric used - choose from [g2, logscore, brier]",
    type=str,
    default='g2',
)

args = parser.parse_args()

logdir = args.logdir
if logdir[-1] != "/":
    logdir = logdir + "/"

############################################################
################ Load Model Data  ##########
complete_data = load_obj("complete_data", logdir)
if "n_splits" not in complete_data.keys():
    complete_data["n_splits"] = 3
ps = dict()
for fold_index in range(complete_data["n_splits"]):
    ps[fold_index] = load_obj("ps_%s" % str(fold_index), logdir)
Ds = dict()
for fold_index in range(complete_data["n_splits"]):
    print('Fold %d'%fold_index)
    Ds[fold_index] = get_scores(
        ps[fold_index],
        complete_data[fold_index],
        args.nsim_ppp,
        args.scr_metric
        )
###########################################################
############### Compare CV scores  ##########
score_names = ['g2', 'logscore', 'brier']
for name in score_names:
    a = [Ds[fold][name] for fold in range(3)]
    print("\n%s Fold Sum %.2f" %(name, np.sum(a)))
