import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from scipy.stats import norm, multivariate_normal, invwishart, invgamma
from numpy.linalg import det, inv

from codebase.file_utils import save_obj, load_obj
from codebase.data import gen_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("sim_case", help="simulation case number", type=int, default=0)
parser.add_argument("stan_model", help="0:full model, 1:no u's, 2: no u's no approx zero betas ", type=int, default=0)
# Optional arguments
parser.add_argument("-nll","--noisy_loadings_level", help="option level for cross loading magnitude", type=int, default=3)
parser.add_argument("-num_chains","--num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("-bnsim","--btstr_nsim", help="random seed for data generation", type=int, default=20)
parser.add_argument("-nd","--nsim_data", help="data size", type=int, default=500)
parser.add_argument("-off", "--standardize", help="standardize the data", type=int, default=1)
parser.add_argument("-th", "--task_handle", help="hande for task", type=str, default="_")
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=0)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
    type=str, default=None)

args = parser.parse_args()


def ff2(yy, model_mu, model_Sigma, p):
    sample_S = np.cov(yy, rowvar=False)
    ldS = np.log(det(sample_S))
    iSigma = inv(model_Sigma)
    ldSigma = np.log(det(model_Sigma))
    n_data = yy.shape[0]
    ff2 =(n_data-1)*(ldSigma+np.sum(np.diag(sample_S @ iSigma))-ldS-p)
    return ff2


def compute_D(mcmc_iter, data, pred=True):
    J = ps['alpha'][mcmc_iter].shape[0]
    if pred == True:
        y_pred=multivariate_normal.rvs(mean= ps['alpha'][mcmc_iter],
                        cov=ps['Marg_cov'][mcmc_iter],
                       size = data['yy'].shape[0])
        return ff2(y_pred, ps['alpha'][mcmc_iter], ps['Marg_cov'][mcmc_iter],
            p = J)

    else:
        return ff2(data['yy'], ps['alpha'][mcmc_iter], ps['Marg_cov'][mcmc_iter],
            p = J)

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
    elif args.stan_model == 4 :
        with open('./codebase/stan_code/cont/EFA/model1.stan', 'r') as file:
            model_code = file.read()
        param_names = ['Marg_cov', 'beta', 'sigma', 'alpha', 'Theta']
    elif args.stan_model == 5 :
        with open('./codebase/stan_code/cont/EFA/model2.stan', 'r') as file:
            model_code = file.read()
        param_names = ['Marg_cov',  'beta', 'sigma', 'alpha', 'Theta', 'Omega']
    else:
        print("Choose stan model {0:benchmark saturated model," \
            "1 CFA/4 EFA:exact zeros no u's, 2 CFA/5 EFA: full factor model}")

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
    elif args.stan_model == 4 :
        param_names = ['Marg_cov', 'beta', 'sigma', 'alpha', 'Theta']
    elif args.stan_model == 5 :
        param_names = ['Marg_cov',  'beta', 'sigma', 'alpha', 'Theta', 'Omega']
    else:
        print("Choose stan model {0 :benchmark saturated model," \
            "1 CFA/4 EFA:exact zeros no u's, 2 CFA/5 EFA: full factor model}")


############################################################
################ Create Data or Load ##########

print("\n\nInitializing results array of length %s"%args.btstr_nsim)

bstr_results = np.empty(args.btstr_nsim)

for iter_k in range(args.btstr_nsim):
    print("\n\nGenerating Continuous data for case %s"%args.sim_case)
    if args.sim_case == 0 :
        data = gen_data(args.nsim_data, off_diag_residual=False,
            random_seed = iter_k)
    elif args.sim_case == 1 :
        data = gen_data(args.nsim_data, off_diag_residual=True,
            random_seed = iter_k)
    elif args.sim_case == 2 :
        data = gen_data(args.nsim_data, noisy_loadings=True, off_diag_residual=False,
            noisy_loadings_level = args.noisy_loadings_level, random_seed = iter_k)
    elif args.sim_case == 3 :
        data = gen_data(args.nsim_data, noisy_loadings=True, off_diag_residual=True,
            noisy_loadings_level = args.noisy_loadings_level, random_seed = iter_k)
    else:
        print("Choose simulation case {0:diag Theta, \
            1:Theta with 6 off diag elements \
            2:Noisy loadings}")


    data['sigma_prior'] = np.diag(np.linalg.inv(np.cov(data['y'], rowvar=False)))
    print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))

    stan_data = dict(N = data['N'], K = data['K'], J = data['J'],
        yy = data['y'], sigma_prior = data['sigma_prior'])

    save_obj(data, 'data_seed'+str(iter_k), log_dir)

    ############################################################
    ################ Fit Model ##########
    print("\n\nFitting model.... \n\n")

    fit_run = sm.sampling(data=stan_data,
        iter=args.num_samples + args.num_warmup,
        warmup=args.num_warmup, chains=args.num_chains,
        init = 0)

    stan_samples= fit_run.extract(permuted=False, pars=param_names)  # return a dictionary of arrays

    if args.num_chains ==1:
        ps = dict()
        for name in param_names:
            ps[name] = np.squeeze(stan_samples[name])
    else:
        ps = stan_samples


    mcmc_length = ps['alpha'].shape[0]
    Ds = np.empty((mcmc_length,2))
    for mcmc_iter in range(mcmc_length):
        Ds[mcmc_iter,0] = compute_D(mcmc_iter, stan_data, pred=False)
        Ds[mcmc_iter,1] = compute_D(mcmc_iter, stan_data, pred=True)


    result =np.sum(Ds[:,0] < Ds[:,1]) / mcmc_length
    bstr_results[iter_k] = result
    print("\n\n\n#######\n\nPPP = %d %%\n\n\n"%np.round(100*result,0))

    np.save(log_dir+"bstr_results", bstr_results)
    save_obj(ps, 'ps_seed'+str(iter_k), log_dir)


print("\n\n\n####### Final Average #######\n\nAverage PPP = %d %%\n\n\n"%np.round(100*np.mean(bstr_results),0))

np.save(log_dir+"bstr_results", bstr_results)
