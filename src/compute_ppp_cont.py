import numpy as np
import pandas as pd
import pystan
import datetime
from tqdm import tqdm
from codebase.model_fit_cont import compute_D
import sys
import os
from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)
args = parser.parse_args()

log_dir = args.logdir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"


data = load_obj("stan_data", log_dir)
ps = load_obj('ps', log_dir)


cn = 0
mcmc_length = ps['alpha'].shape[0]
Ds = np.empty((mcmc_length, 2))
for mcmc_iter in tqdm(range(mcmc_length)):
    Ds[mcmc_iter, 0] = compute_D(data, ps, mcmc_iter, cn, pred=False)
    Ds[mcmc_iter, 1] = compute_D(data, ps, mcmc_iter, cn, pred=True)


result = np.round(100*(np.sum(Ds[:, 0] < Ds[:, 1]) / mcmc_length), 0)
print("PPP = %d %%" % result)
