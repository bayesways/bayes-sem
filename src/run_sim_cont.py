import numpy as np
import pandas as pd
import pystan
import datetime
import os
from codebase.file_utils import save_obj, load_obj
from codebase.data import gen_data
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument(
    "num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument(
    "sim_case", help="simulation case number", type=int, default=0)
parser.add_argument(
    "stan_model", help="0:full model, 1:no u's, 2: no u's no approx zero betas ", type=int, default=0)
# Optional arguments
parser.add_argument("-cv", "--ppp_cv",
                    help="run PPP or CV", type=str, default='ppp')
parser.add_argument("-lm", "--load_model",
                    help="load model", type=bool, default=False)
parser.add_argument("-odr", "--off_diag_residual",
                    help="off_diag_residual", type=bool, default=False)
parser.add_argument("-gd", "--gen_data",
                    help="gen fresh data", type=bool, default=False)
parser.add_argument("-off", "--standardize",
                    help="standardize the data", type=int, default=1)
parser.add_argument("-rho", "--rho_param",
                    help="off diag correlation of Theta", type=float, default=0.1)
parser.add_argument("-num_chains", "--num_chains",
                    help="number of MCMC chains", type=int, default=1)
parser.add_argument("-seed", "--random_seed",
                    help="random seed for data generation", type=int, default=0)
parser.add_argument("-c", "--c_param",
                    help="fixed variances of Theta", type=float, default=1)
parser.add_argument("-nd", "--nsim_data", help="data size",
                    type=int, default=1000)
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="_")
parser.add_argument("-prm", "--print_model",
                    help="print model on screen", type=int, default=0)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
                    type=str, default=None)
parser.add_argument("-sqz", "--squeeze_ps",
                    help="squeeze posterior samples vectors", type=int, default=0)
parser.add_argument("-nfl", "--n_splits",
                    help="number of folds", type=int, default=3)


args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')  # ISO 8601 format
    log_dir = "./log/"+nowstr+"%s_s%sm%s/" % (args.task_handle, args.sim_case,
                                              args.stan_model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir + "/"

############################################################
################ Create Data or Load ##########
if args.existing_directory is None or args.gen_data == True:

    print("\n\nGenerating Continuous data for case")

    print("\n\nGenerating Continuous data for case %s"%args.sim_case)
    if args.sim_case == 0 :
        data = gen_data(args.nsim_data, off_diag_residual=False,
            random_seed = args.random_seed)
    elif args.sim_case == 1 :
        data = gen_data(args.nsim_data, off_diag_residual=True,
            random_seed = args.random_seed)
    elif args.sim_case == 2 :
        data = gen_data(args.nsim_data, cross_loadings=True, off_diag_residual=False,
            cross_loadings_level = 3, random_seed = args.random_seed)
    elif args.sim_case == 3 :
        data = gen_data(args.nsim_data, cross_loadings=True, off_diag_residual=True,
            cross_loadings_level = 3, random_seed = args.random_seed)
    else:
        print("Choose simulation case {0:diag Theta, \
            1:Theta with 6 off diag elements \
            2:Noisy loadings}")

    if args.ppp_cv == 'ppp':  # run PPP
        print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))
        data['sigma_prior'] = np.diag(np.linalg.inv(np.cov(data['y'], rowvar=False)))
        stan_data = dict(N = data['N'], K = data['K'], J = data['J'],
            yy = data['y'], sigma_prior = data['sigma_prior'])
        print("\n\nSaving data to directory %s"% log_dir)
        save_obj(stan_data, 'stan_data', log_dir)
        save_obj(data, 'data', log_dir)


    elif args.ppp_cv == 'cv':  # run CV
        print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))
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

        print("\n\nSaving data folds at %s" % log_dir)
        save_obj(stan_data, 'stan_data', log_dir)
        save_obj(complete_data, 'complete_data', log_dir)
        save_obj(data, 'data', log_dir)
    else:
        print("-cv needs to be 'ppp' or 'cv'")

else:
    print("\n\nReading data from directory %s" % log_dir)
    stan_data = load_obj("stan_data", log_dir)


############################################################
################ Compile Model or Load ##########
if args.load_model == False:

    print("\n\nReading Stan Code from model %d" % args.stan_model)
    if args.stan_model == 0 :
        with open('./codebase/stan_code/cont/EFA/model0.stan', 'r') as file:
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
    try:
        print("\n\nSaving compiled model in directory %s" % log_dir)
        save_obj(sm, 'sm', log_dir)
    except:
        # Print error message
        print("could not save the stan model")
else:
    print("\n\nReading existing compiled model from directory %s" % log_dir)
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
################ Fit Model ##########

if args.ppp_cv == 'ppp':  # run PPP
    print("\n\nFitting model.... \n\n")

    fit_run = sm.sampling(data=stan_data,
                          iter=args.num_samples + args.num_warmup,
                          warmup=args.num_warmup, chains=args.num_chains,
                          n_jobs=4, control={'max_treedepth': 10, 'adapt_delta': 0.9},
                          init = 0)

    try:
        print("\n\nSaving fitted model in directory %s" % log_dir)
        save_obj(fit_run, 'fit', log_dir)
    except:
        # Print error message
        print("could not save the fit object")

    print("\n\nSaving posterior samples in %s" % log_dir)
    # return a dictionary of arrays
    stan_samples = fit_run.extract(permuted=False, pars=param_names)

    if (args.num_chains == 1) and args.squeeze_ps:
        ps = dict()
        for name in param_names:
            ps[name] = np.squeeze(stan_samples[name])
    else:
        ps = stan_samples
    save_obj(ps, 'ps', log_dir)

elif args.ppp_cv == 'cv':  # run CV
    print("\n\nKfold Fitting starts.... \n\n")

    fit_runs = dict()
    for fold_index in range(args.n_splits):
        print("\n\nFitting model.... \n\n")

        fit_runs[fold_index] = sm.sampling(data=stan_data[fold_index],
                                           iter=args.num_samples + args.num_warmup,
                                           warmup=args.num_warmup, chains=args.num_chains,
                                           n_jobs=4, control={'max_treedepth': 15, 'adapt_delta': 0.99})
        # init = 0)
        try:
            print("\n\nSaving fitted model in directory %s" % log_dir)
            save_obj(fit_runs, 'fit', log_dir)
        except:
            # Print error message
            print("could not save the fit object")

    print("\n\nSaving posterior samples in %s ..." % log_dir)

    stan_samples = dict()
    for fold_index in range(args.n_splits):
        print("\n\nSaving posterior for fold %s samples in %s" %
              (fold_index, log_dir))
        # return a dictionary of arrays
        stan_samples[fold_index] = fit_runs[fold_index].extract(permuted=False,
                                                                pars=param_names)

        if (args.num_chains == 1) and args.squeeze_ps:
            ps = dict()
            for name in param_names:
                ps[name] = np.squeeze(stan_samples[fold_index][name])
        else:
            ps = stan_samples[fold_index]
        save_obj(ps, 'ps_'+str(fold_index), log_dir)

else:
    print("-cv needs to be 'ppp' or 'cv'")
