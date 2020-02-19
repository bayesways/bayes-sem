import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os

from codebase.file_utils import save_obj, load_obj
from codebase.data_db import gen_data4
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("sim_case", help="simulation case number", type=int, default=0)
# Optional arguments
parser.add_argument("-lm","--load_model", help="load model", type=bool, default=False)
parser.add_argument("-odr","--off_diag_residual", help="off_diag_residual", type=bool, default=False)
parser.add_argument("-gd","--gen_data", help="gen fresh data", type=bool, default=False)
parser.add_argument("-rho","--rho_param", help="off diag correlation of Theta", type=float, default=0.3)
parser.add_argument("-num_chains","--num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("-seed","--random_seed", help="random seed for data generation", type=int, default=0)
parser.add_argument("-c","--c_param", help="fixed variances of Theta", type=float, default=0.2)
parser.add_argument("-nd","--nsim_data", help="data size", type=int, default=1000)
parser.add_argument("-th", "--task_handle", help="hande for task", type=str, default="_")
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=0)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
    type=str, default=None)

args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
    log_dir =  "./log/"+nowstr+"%s/"%(args.task_handle)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir+ "/"

############################################################
################ Create Data or Load ##########
if args.existing_directory is None or args.gen_data == True:

    print("\n\nGenerating Continuous data for case")

    data = gen_data4(args.nsim_data,
        c = args.c_param,
        rho = args.rho_param,
        off_diag_residual = args.off_diag_residual,
        random_seed = args.random_seed)

    stan_data = dict(
        N = data['N'],
        J = data['J'],
        K = data['K'],
        DD = data['D'],
        c = data['c']
        )
    print("\n\nSaving data to directory %s"% log_dir)
    save_obj(stan_data, 'stan_data', log_dir)
    save_obj(data, 'data', log_dir)

else:
    print("\n\nReading data from directory %s"%  log_dir)
    stan_data = load_obj("stan_data", log_dir)


############################################################
################ Compile Model or Load ##########
if args.load_model == False:

    if args.sim_case == 1 :
        with open('./codebase/stan_code/discr/CFA/logit/test/model4.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']

    elif args.sim_case == 2 :
        with open('./codebase/stan_code/discr/CFA/logit/test/model4b.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
            'Omega_cov', 'Phi_cov']
    elif args.sim_case == 3 :
        with open('./codebase/stan_code/discr/CFA/logit/test/model5.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta',  'Marg_cov', 'Omega_cov',
            'Omega_corr', 'sigma_omega']

    else:
        print('model is 1,2,3')

    if bool(args.print_model):
        print(model_code)
    file = open(log_dir+"model.txt", "w")
    file.write(model_code)
    file.close()


    if bool(args.print_model):
        print(model_code)
    file = open(log_dir+"model.txt", "w")
    file.write(model_code)
    file.close()



    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)

    try:
        print("\n\nSaving compiled model in directory %s"%log_dir)
        save_obj(sm, 'sm', log_dir)
    except:
        # Print error message
        print("could not save the stan model")
else:
    print("\n\nReading existing compiled model from directory %s"%log_dir)
    sm = load_obj('sm', log_dir)

    if args.sim_case == 1 :
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']
    elif args.sim_case == 2 :
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov', 'Phi_cov']
    elif args.sim_case == 3 :
        param_names = ['alpha', 'yy',  'beta',  'Marg_cov', 'Omega_cov',
            'Omega_corr', 'sigma_omega']
    else:
        print('model is 1,2,3')

############################################################
################ Fit Model ##########
print("\n\nFitting model.... \n\n")


fit_run = sm.sampling(data=stan_data,
    iter=args.num_samples + args.num_warmup,
    warmup=args.num_warmup, chains=args.num_chains)
    # init = 0)
    # control = {'max_treedepth':15, 'adapt_delta':0.99})

try:
    print("\n\nSaving fitted model in directory %s"%log_dir)
    save_obj(fit_run, 'fit', log_dir)
except:
    # Print error message
    print("could not save the fit object")


print("\n\nSaving posterior samples in %s"%log_dir)
stan_samples= fit_run.extract(permuted=False, pars=param_names)  # return a dictionary of arrays

if args.num_chains ==1:
    ps = dict()
    for name in param_names:
        ps[name] = np.squeeze(stan_samples[name])
else:
    ps = stan_samples
save_obj(ps, 'ps', log_dir)
