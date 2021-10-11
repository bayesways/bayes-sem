import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from codebase.file_utils import save_obj, load_obj
from codebase.data import gen_data_binary
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument(
    "num_samples", help="number of post-warm up iterations", type=int, default=1000)
# Optional arguments
parser.add_argument("-cm", "--compile_model",
                    help="load model", type=int, default=0)
parser.add_argument("-num_chains", "--num_chains",
                    help="number of MCMC chains", type=int, default=4)
parser.add_argument("-nd", "--nsim_data", help="data size",
                    type=int, default=2000)
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="_")
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
                    type=str, default=None)
parser.add_argument("-sqz", "--squeeze_ps",
                    help="squeeze posterior samples vectors", type=int, default=0)

args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')  # ISO 8601 format
    log_dir = "./log/"+nowstr+"%s/" % (args.task_handle)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir + "/"

############################################################
################ Compile Model or Load ##########
path_to_stan = './codebase/stan_code/discr/'

# alternative parametrisation of model 1
with open(path_to_stan+'CFA/logit/model1_B.stan', 'r') as file:
    model_code = file.read()
param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']

print("\n\nCompiling model")
sm = pystan.StanModel(model_code=model_code, verbose=False)
print("\n\nSaving compiled model in directory %s" % log_dir)
save_obj(sm, 'sm', log_dir)


############################################################
################ Create Data or Load ##########

for random_seed in range(2):
    data = gen_data_binary(
        args.nsim_data,
        random_seed=random_seed
        )
    stan_data = dict(
        N=data['N'],
        J=data['J'],
        K=data['K'],
        DD=data['D']
    )
    print("\n\nSaving data to directory %s" % log_dir)
    save_obj(stan_data, 'stan_data'+str(random_seed), log_dir)
    save_obj(data, 'data'+str(random_seed), log_dir)

############################################################
################ Fit Model ##########

    fit_run = sm.sampling(
        data=stan_data,
        iter=args.num_samples + args.num_warmup,
        warmup=args.num_warmup, chains=args.num_chains,
        n_jobs=4,
        control={'max_treedepth': 15, 'adapt_delta': 0.99},
        init = 0
        )
    print("\n\nSaving posterior samples in %s" % log_dir)
    # return a dictionary of arrays
    stan_samples = fit_run.extract(permuted=False, pars=param_names)

    if (args.num_chains == 1) and args.squeeze_ps:
        ps = dict()
        for name in param_names:
            ps[name] = np.squeeze(stan_samples[name])
    else:
        ps = stan_samples
    save_obj(ps, 'ps'+str(random_seed), log_dir)
