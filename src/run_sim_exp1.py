import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os

from codebase.file_utils import save_obj, load_obj
from codebase.data import gen_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("sim_case", help="simulation case number", type=int, default=0)
parser.add_argument("stan_model", help="0:full model, 1:no u's, 2: no u's no approx zero betas ", type=int, default=0)
# Optional arguments
parser.add_argument("-num_chains","--num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("-off", "--standardize", help="standardize the data", type=int, default=1)
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
    print("\n\nGenerating data for case %s"%args.sim_case)
    if args.sim_case == 0 :
        data = gen_data(500, off_diag_residual=False)
    elif args.sim_case == 1 :
        data = gen_data(500, off_diag_residual=True)
    else:
        print("Choose simulation case {0:diag Theta, 1:Theta with 6 off diag elements}")


    data['sigma_prior'] = np.diag(np.linalg.inv(np.cov(data['y'], rowvar=False)))
    print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))

    stan_data = dict(N = data['N'], K = data['K'], J = data['J'],
        yy = data['y'], sigma_prior = data['sigma_prior'])
    print("\n\nSaving data to directory %s"% log_dir)
    save_obj(stan_data, 'stan_data', log_dir)
    save_obj(data, 'data', log_dir)

else:
    print("\n\nReading data from directory %s"%  log_dir)
    stan_data = load_obj("stan_data", log_dir)


############################################################
################ Compile Model or Load ##########
if args.existing_directory is None:

    print("\n\nReading Stan Code from model %d" % args.stan_model)
    if args.stan_model == 0 :
        with open('./codebase/stan_code/cont/CFA/model0.stan', 'r') as file:
            model_code = file.read()
        param_names = ['Marg_cov', 'alpha']
    elif args.stan_model == 1 :
        with open('./codebase/stan_code/cont/CFA/model1.stan', 'r') as file:
            model_code = file.read()
        param_names = ['Marg_cov', 'beta', 'Phi_cov', 'sigma', 'alpha', 'Theta']
    elif args.stan_model == 2 :
        with open('./codebase/stan_code/cont/CFA/model2_sim.stan', 'r') as file:
            model_code = file.read()
        param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma', 'alpha',
            'Theta', 'Omega']
    else:
        print("Choose stan model {0:benchmark saturated model, 1:exact zeros no u's, 2: full factor model}")

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

    if args.stan_model == 0 :
        param_names = ['Marg_cov', 'alpha']
    elif args.stan_model == 1 :
        param_names = ['Marg_cov', 'beta', 'Phi_cov', 'sigma', 'alpha', 'Theta']
    elif args.stan_model == 2 :
        param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma', 'alpha',
            'Theta', 'Omega']
    else:
        print("Choose stan model {0:benchmark saturated model, 1:exact zeros no u's, 2: full factor model}")


############################################################
################ Fit Model ##########
print("\n\nFitting model.... \n\n")

fit_run = sm.sampling(data=stan_data,
    iter=args.num_samples + args.num_warmup,
    warmup=args.num_warmup, chains=args.num_chains)

print("\n\nSaving fitted model in directory %s"%log_dir)
save_obj(fit_run, 'fit', log_dir)


print("\n\nSaving posterior samples in %s"%log_dir)


stan_samples= fit_run.extract(permuted=False, pars=param_names)  # return a dictionary of arrays

if args.num_chains ==1:
    ps = dict()
    for name in param_names:
        ps[name] = np.squeeze(stan_samples[name])
else:
    ps = stan_samples
save_obj(ps, 'ps', log_dir)
