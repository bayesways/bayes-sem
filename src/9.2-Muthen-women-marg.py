import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, invwishart, invgamma
from statsmodels.tsa.stattools import acf
import datetime
import sys
import os

from codebase.file_utils import save_obj, load_obj
from codebase.post_process import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("num_chains", help="number of MCMC chains", type=int, default=1)
# Optional arguments
parser.add_argument("-th", "--task_handle", help="hande for task", type=str, default="_")
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
    type=str, default=None)

args = parser.parse_args()


df = pd.read_csv("../dat/muthen_women.csv")


data = dict()
data['N'] = df.shape[0]
data['K'] = 5
data['J'] = df.shape[1]
data['y'] = df.values


stan_data = dict(N = data['N'], K = data['K'], J = data['J'], yy = data['y'])

with open('./codebase/stan_code/cont/CFA/marg_m.stan', 'r') as file:
    model_code = file.read()
print(model_code)


if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
    log_dir =  "./log/"+nowstr+"%s/" % args.task_handle
    sm = pystan.StanModel(model_code=model_code, verbose=False)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_obj(sm, 'sm', log_dir)

else:
    log_dir = "./log/"+args.existing_directory
    sm = load_obj('sm', log_dir)

fit_run = sm.sampling(data=stan_data,
    iter=args.num_samples + args.num_warmup,
    warmup=args.num_warmup, chains=args.num_chains)

save_obj(fit_run, 'fit', log_dir)
