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
parser.add_argument(
    "-nsim",
    "--nsim_ppp",
    help="number of posterior samples to use for PPP",
    type=int,
    default=None,
)
parser.add_argument(
    "-scr",
    "--scr_method",
    help="method used for score",
    type=int,
    default=1,
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
Ds = np.empty(3)
for fold_index in range(complete_data["n_splits"]):
    Ds[fold_index] = get_lgscr(
        ps[fold_index],
        complete_data[fold_index],
        args.nsim_ppp,
        args.scr_method
        )

###########################################################
############### Compare CV scores  ##########
print("\nFold Sum %.2f" % np.sum(Ds))
for f in range(3):
    print("Fold %.2f" % Ds[f])
