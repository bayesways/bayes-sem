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
parser.add_argument(
    "sim_case", help="simulation case number", type=int, default=0)
parser.add_argument(
    "stan_model", help="0:full model, 1:no u's, 2: no u's no approx zero betas ", type=int, default=0)
# Optional arguments
parser.add_argument("-cv", "--ppp_cv",
                    help="run PPP or CV", type=str, default='ppp')
parser.add_argument("-cm", "--compile_model",
                    help="load model", type=int, default=0)
parser.add_argument("-odr", "--off_diag_residual",
                    help="off_diag_residual", type=bool, default=False)
parser.add_argument("-gd", "--gen_data",
                    help="gen fresh data", type=int, default=1)
parser.add_argument("-rho", "--rho_param",
                    help="off diag correlation of Theta", type=float, default=0.1)
parser.add_argument("-num_chains", "--num_chains",
                    help="number of MCMC chains", type=int, default=4)
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
parser.add_argument("-cv_seed", "--cv_seed",
                    help="random seed for CV", type=int, default=34)


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
if args.gen_data == 1:

    print("\n\nGenerating Continuous data for case")

    if args.sim_case == 0:
        data = gen_data_binary(
            args.nsim_data,
            c=args.c_param,
            off_diag_residual=False,
            random_seed=args.random_seed
            )
    elif args.sim_case == 1:
        data = gen_data_binary(
            args.nsim_data,
            rho2=args.rho_param,
            c=args.c_param,
            off_diag_residual=True,
            random_seed=args.random_seed
            )
    elif args.sim_case == 2:
        data = gen_data_binary(
            args.nsim_data,
            c=args.c_param,
            off_diag_residual=False,
            cross_loadings=True,
            cross_loadings_level=3,
            random_seed=args.random_seed
            )
    else:
        print("Choose simulation case 0:Clean data ")
        print("Choose simulation case 1:Off-diag residuals")
        print("Choose simulation case 2:Cross loadings")

    if args.ppp_cv == 'ppp':  # run PPP
        stan_data = dict(
            N=data['N'],
            J=data['J'],
            K=data['K'],
            DD=data['D']
        )
        print("\n\nSaving data to directory %s" % log_dir)
        save_obj(stan_data, 'stan_data', log_dir)
        save_obj(data, 'data', log_dir)
    elif args.ppp_cv == 'cv':  # run CV
        X = data['D']
        kf = KFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.cv_seed
            )
        kf.get_n_splits(X)

        stan_data = dict()
        complete_data = dict()
        fold_index = 0
        for train_index, test_index in kf.split(X):
            data_fold = dict()
            data_fold['D_train'], data_fold['D_test'] = X[train_index], X[test_index]
            data_fold['N_train'], data_fold['N_test'] = data_fold['D_train'].shape[0], data_fold['D_test'].shape[0]
            stan_data[fold_index] = dict(N=data_fold['N_train'],
                                         K=data['K'],
                                         J=data['J'],
                                         DD=data_fold['D_train'])
            test_data_fold = dict(N=data_fold['N_test'],
                                  K=data['K'],
                                  J=data['J'],
                                  DD=data_fold['D_test'])
            complete_data[fold_index] = dict(
                train=stan_data[fold_index], test=test_data_fold)

            fold_index += 1

        print("\n\nSaving data folds at %s" % log_dir)
        complete_data['cv_seed'] = args.cv_seed
        complete_data['n_splits'] = args.n_splits
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
path_to_stan = './codebase/stan_code/discr/'

print("\n\nReading Stan Code from model %d" % args.stan_model)
if args.stan_model == 1 :
    with open(path_to_stan+'CFA/logit/model1.stan', 'r') as file:
        model_code = file.read()
    param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
elif args.stan_model == 2 :
    with open(path_to_stan+'CFA/logit/model2.stan', 'r') as file:
        model_code = file.read()
    param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
        'Omega_cov', 'Phi_cov']
elif args.stan_model == 5 :
    with open(path_to_stan+'EFA/model1.stan', 'r') as file:
        model_code = file.read()
    param_names = ['beta', 'alpha', 'zz', 'yy']
elif args.stan_model == 6 :
    with open(path_to_stan+'EFA/model2.stan', 'r') as file:
        model_code = file.read()
    param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']
else:
    print('model is 1:6')

if bool(args.print_model):
    print(model_code)
file = open(log_dir+"model.txt", "w")
file.write(model_code)
file.close()

if args.compile_model==0:
    with open('log/compiled_models/discr/model%s/model.txt' % args.stan_model, 'r') as file:
        saved_model = file.read()
    if saved_model == model_code:
        sm = load_obj('sm', 'log/compiled_models/discr/model%s/' % args.stan_model)
        if args.stan_model == 1:
            param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
        elif args.stan_model == 2:
            param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
                'Omega_cov', 'Phi_cov']
        elif args.stan_model == 5:
            param_names = ['beta', 'alpha', 'zz', 'yy']
        elif args.stan_model == 6:
            param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']
        else:
            print("model option should be in [0,1,2,3]")

else:
    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)
    try:
        print("\n\nSaving compiled model in directory %s" % log_dir)
        save_obj(sm, 'sm', 'log/compiled_models/discr/model%s/' % args.stan_model)
        file = open('log/compiled_models/discr/model%s/model.txt' %
                    args.stan_model, "w")
        file.write(model_code)
        file.close()
    except:
        print("Couldn't save model in model bank")

print("\n\nSaving compiled model in directory %s" % log_dir)
save_obj(sm, 'sm', log_dir)


############################################################
################ Fit Model ##########

if args.ppp_cv == 'ppp':  # run PPP
    print("\n\nFitting model.... \n\n")

    fit_run = sm.sampling(
        data=stan_data,
        iter=args.num_samples + args.num_warmup,
        warmup=args.num_warmup, chains=args.num_chains,
        n_jobs=4,
        control={'max_treedepth': 15, 'adapt_delta': 0.99},
        init = 0
        )

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

        fit_runs[fold_index] = sm.sampling(
            data=stan_data[fold_index],
            iter=args.num_samples + args.num_warmup,
            warmup=args.num_warmup, chains=args.num_chains,
            n_jobs=4,
            control={'max_treedepth': 15, 'adapt_delta': 0.99},
            init = 0
            )
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
