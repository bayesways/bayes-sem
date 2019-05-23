import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from scipy.stats import norm, multivariate_normal
from codebase.data import gen_data

from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("data_dir", help="directory to data", type=str, default=None)

# Optional arguments
parser.add_argument("-rs", "--random_seed", help="random seed to use for data generation", type=int, default=None)
parser.add_argument("-th", "--task_handle", help="handle (name) for task", type=str, default="_")
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=0)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
    type=str, default=None)

args = parser.parse_args()

print("\n\nLoading data")
data = load_obj('data', args.data_dir)



if args.existing_directory is None:
    print("\n\nCreating directory")
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
    log_dir =  "./log/"+nowstr+"%s/" % args.task_handle

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open('./codebase/stan_code/cont/CFA/marg_simulation.stan', 'r') as file:
        model_code = file.read()
    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)
    print("\n\nSaving compiled model in directory %s"%log_dir)
    save_obj(sm, 'sm', log_dir)

else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir+ "/"
    print("\n\nLoading existing data from %s"%log_dir)
    data = load_obj('data', log_dir)
    print("\n\nReading existing compiled model from directory %s"%log_dir)
    sm = load_obj('sm', log_dir)


stan_data = dict(N = data['N'], K = data['K'], J = data['J'], yy = data['y'])

if bool(args.print_model):
    print(model_code)

print("\n\nFitting model.... \n\n")

fit_run = sm.sampling(data=stan_data,
    iter=args.num_samples + args.num_warmup,
    warmup=args.num_warmup, chains=args.num_chains)

print("\n\nSaving fitted model in directory %s"%log_dir)
save_obj(fit_run, 'fit', log_dir)

print("\n\nSaving posterior samples in %s"%log_dir)
param_names = ['Omega_beta', 'beta', 'V_corr', 'V' , 'alpha', 'sigma', 'sigma_z', 'uu']

stan_samples= fit_run.extract(permuted=False, pars=param_names)  # return a dictionary of arrays

if args.num_chains ==1:
    ps = dict()
    for name in param_names:
        ps[name] = np.squeeze(stan_samples[name])
else:
    ps = stan_samples
save_obj(ps, 'ps', log_dir)
