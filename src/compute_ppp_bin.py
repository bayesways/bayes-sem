import numpy as np
import pandas as pd
import pystan
import os
from codebase.file_utils import save_obj, load_obj
from codebase.model_fit_bin import get_PPP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument(
    "-nsim", "--nsim_ppp",
    help="number of posterior samples to use for PPP",
    type=int, default=100
    )

args = parser.parse_args()

log_dir = args.logdir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"


data = load_obj('data', log_dir)
ps = load_obj('ps', log_dir)
num_chains = ps['alpha'].shape[1]
num_samples = ps['alpha'].shape[0]


ppp_cn = np.empty(num_chains)
for cn in range(num_chains):
    PPP_vals = get_PPP(data, ps, cn, args.nsim_ppp)

    ppp_cn[cn] = np.sum(PPP_vals[:, 0] < PPP_vals[:, 1])/args.nsim_ppp
    print(ppp_cn[cn])

print("Avg PPP %.2f"%ppp_cn.mean())
