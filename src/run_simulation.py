import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from codebase.data import gen_data

from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("num_chains", help="number of MCMC chains", type=int, default=1)
# Optional arguments
parser.add_argument("-rs", "--random_seed", help="random seed to use for data generation", type=int, default=None)
parser.add_argument("-th", "--task_handle", help="hande for task", type=str, default="_")
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=0)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
    type=str, default=None)

args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
    log_dir =  "./log/"+nowstr+"%s/" % args.task_handle
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir+ "/"

############################################################
################ Create Data or Load ##########
if args.existing_directory is None:
    print("\n\nGenerating data")
    data = gen_data(500, 6, 2, random_seed = args.random_seed)
    print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))
    data['sigma_prior'] = np.diag(np.linalg.inv(np.cov(data['y'], rowvar=False)))

    # print("\n\n********** \nReplacing last 10% of data with 10's \n***********\n\n")
    # data['y'][-50:] = 3
    print("\nSaving data\n\n")
    save_obj(data, 'data', log_dir)


    stan_data = dict(N = data['N'], K = data['K'], J = data['J'],
        yy = data['y'], sigma_prior = data['sigma_prior'])
    print("\n\nSaving data to directory %s"% log_dir)
    save_obj(stan_data, 'stan_data', log_dir)

else:
    print("\n\nReading data from directory %s"%  log_dir)
    stan_data = load_obj("stan_data", log_dir)


############################################################
################ Compile Model or Load ##########
if args.existing_directory is None:

    with open('./codebase/stan_code/cont/CFA/marg_simulation.stan', 'r') as file:
        model_code = file.read()
    if bool(args.print_model):
        print(model_code)
    file = open(log_dir+"model.txt", "w")
    file.write(model_code)
    file.close()


    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)

    print("\n\nSaving compiled model in directory %s"%log_dir)
    save_obj(sm, 'sm', log_dir)


else:
    print("\n\nReading existing compiled model from directory %s"%log_dir)
    sm = load_obj('sm', log_dir)


############################################################
################ Fit Model ##########
print("\n\nFitting model.... \n\n")

fit_run = sm.sampling(data=stan_data,
    iter=args.num_samples + args.num_warmup,
    warmup=args.num_warmup, chains=args.num_chains)

print("\n\nSaving fitted model in directory %s"%log_dir)
save_obj(fit_run, 'fit', log_dir)


print("\n\nSaving posterior samples in %s"%log_dir)
param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma',
    'sigma_z', 'alpha', "Theta", 'uu', 'Omega', 'Marg_cov2']

stan_samples= fit_run.extract(permuted=False, pars=param_names)  # return a dictionary of arrays

if args.num_chains ==1:
    ps = dict()
    for name in param_names:
        ps[name] = np.squeeze(stan_samples[name])
else:
    ps = stan_samples
save_obj(ps, 'ps', log_dir)
