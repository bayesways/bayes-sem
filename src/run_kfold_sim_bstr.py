import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal

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
parser.add_argument("-bnsim","--btstr_nsim", help="random seed for data generation", type=int, default=10)
parser.add_argument("-nfl", "--n_splits", help="number of folds", type=int, default=3)
parser.add_argument("-num_chains","--num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("-nd","--nsim_data", help="data size", type=int, default=500)
parser.add_argument("-off", "--standardize", help="standardize the data", type=int, default=1)
parser.add_argument("-th", "--task_handle", help="hande for task", type=str, default="_")
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=0)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
    type=str, default=None)

args = parser.parse_args()

def Nlogpdf(yy, mean, cov):
    return multivariate_normal.logpdf(yy, mean, cov, allow_singular=True)

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

    if args.stan_model == 1 :
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

    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)
    save_obj(sm, 'sm', log_dir)

    with open('./codebase/stan_code/cont/CFA/model0.stan', 'r') as file:
        model_code0 = file.read()
    param_names0 = ['Marg_cov', 'alpha']
    sm0 = pystan.StanModel(model_code=model_code0, verbose=False)
    save_obj(sm0, 'sm0', log_dir)

else:
    print("\n\nReading existing compiled model from directory %s"%log_dir)
    sm = load_obj('sm', log_dir)

    sm0 = load_obj('sm0', log_dir)
    param_names0 = ['Marg_cov', 'alpha']

    if args.stan_model == 1 :
        param_names = ['Marg_cov', 'beta', 'Phi_cov', 'sigma', 'alpha', 'Theta']
    elif args.stan_model == 2 :
        param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma', 'alpha',
            'Theta', 'Omega']
    elif args.stan_model == 4 :
        param_names = ['Marg_cov', 'beta', 'sigma', 'alpha', 'Theta']
    elif args.stan_model == 5 :
        param_names = ['Marg_cov',  'beta', 'sigma', 'alpha', 'Theta', 'Omega']
    else:
        print("Choose stan model {0:benchmark saturated model," \
            "1 CFA/4 EFA:exact zeros no u's, 2 CFA/5 EFA: full factor model}")




############################################################
################ Create Data or Load ##########
print("\n\nInitializing results array of length %s"%args.btstr_nsim)

bstr_results = np.empty(args.btstr_nsim)

for iter_k in range(args.btstr_nsim):
    # print("\n\nGenerating Continuous data for case %s"%args.sim_case)
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

    # print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))

    X = data['y']
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=34)
    kf.get_n_splits(X)

    stan_data = dict()
    complete_data = dict()
    fold_index = 0
    for train_index, test_index in kf.split(X):
        data_fold = dict()
        data_fold['y_train'], data_fold['y_test'] = X[train_index], X[test_index]
        data_fold['N_train'], data_fold['N_test'] = data_fold['y_train'].shape[0], data_fold['y_test'].shape[0]
        stan_data[fold_index] = dict(N = data_fold['N_train'],
                                          K = data['K'],
                                          J = data['J'],
                                          yy = data_fold['y_train'],
                                          sigma_prior = np.diag(np.linalg.inv(np.cov(data_fold['y_train'], rowvar=False)))
                                          )
        test_data_fold = dict(N = data_fold['N_test'],
                                          K = data['K'],
                                          J = data['J'],
                                          yy = data_fold['y_test'],
                                          sigma_prior = np.diag(np.linalg.inv(np.cov(data_fold['y_test'], rowvar=False)))
                                          )
        complete_data[fold_index] = dict( train = stan_data[fold_index], test = test_data_fold)

        fold_index += 1

    # print("\n\nSaving data folds at %s"%log_dir)
    save_obj(complete_data, 'complete_data_seed'+str(iter_k), log_dir)


    ############################################################
    ################ Fit Model ##########
    # print("\n\nKfold Fitting starts.... \n\n")

    fit_runs = dict()
    fit_runs0 = dict()
    for fold_index in range(args.n_splits):
        # print("\n\nFitting model.... \n\n")

        fit_runs[fold_index] = sm.sampling(data=stan_data[fold_index],
                iter=args.num_samples + args.num_warmup,
                warmup=args.num_warmup, chains=args.num_chains, init = 0)

        fit_runs0[fold_index] = sm0.sampling(data=stan_data[fold_index],
                iter=args.num_samples + args.num_warmup,
                warmup=args.num_warmup, chains=args.num_chains, init = 0)

    # print("\n\nSaving posterior samples in %s ..."%log_dir)

    stan_samples = dict()
    for fold_index in range(3):
        # print("\n\nSaving posterior for fold %s samples in %s"%(fold_index, log_dir))
        stan_samples[fold_index] = fit_runs[fold_index].extract(permuted=False, pars=param_names)  # return a dictionary of arrays
        for name in param_names:
            stan_samples[fold_index][name] = np.squeeze(stan_samples[fold_index][name])

    stan_samples0 = dict()
    for fold_index in range(3):
        # print("\n\nSaving posterior for fold %s samples in %s"%(fold_index, log_dir))
        stan_samples0[fold_index] = fit_runs0[fold_index].extract(permuted=False, pars=param_names0)  # return a dictionary of arrays
        for name in param_names0:
            stan_samples0[fold_index][name] = np.squeeze(stan_samples0[fold_index][name])

    model_posterior_samples = dict()
    model_posterior_samples[1] = stan_samples0
    model_posterior_samples[2] = stan_samples

    # print("\n\nComputing Folds...\n\n")

    mcmc_length = model_posterior_samples[1][0]['alpha'].shape[0]

    Ds = np.empty((mcmc_length,3))
    for fold_index in range(3):
        for mcmc_iter in range(mcmc_length):
            model_1_lgpdf = Nlogpdf(complete_data[fold_index]['test']['yy'],
                model_posterior_samples[1][fold_index]['alpha'][mcmc_iter],
                model_posterior_samples[1][fold_index]['Marg_cov'][mcmc_iter])


            model_2_lgpdf = Nlogpdf(complete_data[fold_index]['test']['yy'],
                model_posterior_samples[2][fold_index]['alpha'][mcmc_iter],
                model_posterior_samples[2][fold_index]['Marg_cov'][mcmc_iter])
            Ds[mcmc_iter, fold_index] = -2*np.sum(model_1_lgpdf - model_2_lgpdf)

    fold_results = np.sum(Ds>0, axis=0)/mcmc_length
    print("\n\n\n",fold_results)
    result = np.mean(fold_results)

    bstr_results[iter_k] = result
    print("\n\n\n####### Iter %s ######\n\n3-fold Index = %d %%\n\n\n"%(iter_k, np.round(100*result,0)))


print("\n\n\n####### Final Average #######\n\nAverage 3-Fold Index= %d %%\n\n\n"%np.round(100*np.mean(bstr_results),0))

np.save(log_dir+"bstr_results", bstr_results)
