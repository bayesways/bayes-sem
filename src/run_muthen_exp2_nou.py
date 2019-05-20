import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os

from codebase.file_utils import save_obj, load_obj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("gender", help="run men or women", type=str, default="men")
# Optional arguments
parser.add_argument("-th", "--task_handle", help="hande for task", type=str, default="_")
parser.add_argument("-prm", "--print_model", help="print model on screen", type=bool, default=False)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
    type=str, default=None)

args = parser.parse_args()

print("\n\nReading data for %s"%args.gender)
df = pd.read_csv("../dat/muthen_"+args.gender+".csv")
df = df.replace(-9, np.nan).astype(float)
df.dropna(inplace=True)
df = df.astype(int)



data = dict()
data['N'] = df.shape[0]
data['K'] = 5
data['J'] = df.shape[1]
data['y'] = df.values

print("\n\n********** \nReplacing last 10% of data with 10's \n***********\n\n")
data['y'][-int(data['N']*.1):] = 10

print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))

stan_data = dict(N = data['N'], K = data['K'], J = data['J'], yy = data['y'])

with open('./codebase/stan_code/cont/CFA/marg_m_nou.stan', 'r') as file:
    model_code = file.read()
if args.print_model == True:
    print(model_code)


if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') # ISO 8601 format
    log_dir =  "./log/"+nowstr+"%s/" % args.task_handle
    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("\n\nSaving compiled model in directory %s"%log_dir)
    save_obj(sm, 'sm', log_dir)


else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir+ "/"
    print("\n\nReading existing compiled model from directory %s"%log_dir)
    sm = load_obj('sm', log_dir)


print("\n\nFitting model.... \n\n")

fit_run = sm.sampling(data=stan_data,
    iter=args.num_samples + args.num_warmup,
    warmup=args.num_warmup, chains=args.num_chains)

print("\n\nSaving fitted model in directory %s"%log_dir)
save_obj(fit_run, 'fit', log_dir)


print("\n\nSaving posterior samples in %s"%log_dir)
param_names = ['Omega_beta', 'beta', 'V_corr', 'V' , 'alpha', 'sigma', 'sigma_z']

stan_samples= fit_run.extract(permuted=False, pars=param_names)  # return a dictionary of arrays

if args.num_chains ==1:
    ps = dict()
    for name in param_names:
        ps[name] = np.squeeze(stan_samples[name])
else:
    ps = stan_samples
save_obj(ps, 'ps', log_dir)
