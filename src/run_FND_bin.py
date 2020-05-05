import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os

from sklearn.model_selection import KFold
from codebase.file_utils import save_obj, load_obj
from codebase.data_FND import get_FND_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument(
    "num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument(
    "stan_model", help="choose what model to run-see list in code", type=int, default=0)
# Optional arguments
parser.add_argument("-cv", "--ppp_cv",
                    help="run PPP or CV", type=str, default='ppp')
parser.add_argument("-sc", "--sim_case",
                    help="simulation case number", type=int, default=1)
parser.add_argument("-nfac", "--num_factors",
                    help="number of factors for EFA", type=int, default=2)
parser.add_argument("-lm", "--load_model",
                    help="load model", type=bool, default=False)
parser.add_argument("-num_chains", "--num_chains",
                    help="number of MCMC chains", type=int, default=1)
parser.add_argument("-sqz", "--squeeze_ps",
                    help="squeeze posterior samples vectors", type=int, default=0)
parser.add_argument("-seed", "--random_seed",
                    help="random seed for data generation", type=int, default=0)
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="FND")
parser.add_argument("-prm", "--print_model",
                    help="print model on screen", type=int, default=0)
parser.add_argument("-save_stan", "--save_stan",
                    help="print model on screen", type=bool, default=False)
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
                    type=str, default=None)
parser.add_argument("-nfl", "--n_splits",
                    help="number of folds", type=int, default=3)

args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')  # ISO 8601 format
    log_dir = "./log/"+nowstr+"%s_%s_sc%s_m%s_f%s/" % (args.task_handle, args.ppp_cv, args.sim_case,
                                                       args.stan_model, args.num_factors)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir + "/"

############################################################
################ Create Data or Load ##########
if args.existing_directory is None:

    print("\n\nGenerating Continuous data for case")

    if args.sim_case == 1:
        data = get_FND_data()
    elif args.sim_case == 2:
        data = get_FND_data(alt_order=True)
    else:
        print("Only Simulation Option is 1 or 2")

    # define number of factors for EFA only
    if args.stan_model >= 4 and args.stan_model <= 7:
        data['K'] = args.num_factors

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
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=34)
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

    if args.stan_model == 1:
        with open('./codebase/stan_code/discr/CFA/logit/model1.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
    elif args.stan_model == 2:
        with open('./codebase/stan_code/discr/CFA/logit/model2.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
                       'Omega_cov', 'Phi_cov']
    elif args.stan_model == 3:  # alt param of model 2
        with open('./codebase/stan_code/discr/CFA/logit/model2_prm2.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
                       'Omega_cov', 'Phi_cov']
    elif args.stan_model == 4:  # EFA no u's
        with open('./codebase/stan_code/discr/EFA/model1.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'yy']
    elif args.stan_model == 5:  # EFA with u's
        with open('./codebase/stan_code/discr/EFA/model2.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']
    elif args.stan_model == 6:  # EFA lower triang no u's
        with open('./codebase/stan_code/discr/EFA/model1_lower.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'yy']
    elif args.stan_model == 7:  # EFA lower triang with u's
        with open('./codebase/stan_code/discr/EFA/model2_lower.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']
    elif args.stan_model == 8:  # alt param of model 2
        with open('./codebase/stan_code/discr/CFA/logit/model5_n.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Phi_cov']    
    elif args.stan_model == 9:  # alt param of model 2
        with open('./codebase/stan_code/discr/CFA/logit/model5.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
                       'Omega_cov', 'Phi_cov']                           
    else:
        print('model is 1:9')

    if bool(args.print_model):
        print(model_code)
    file = open(log_dir+"model.txt", "w")
    file.write(model_code)
    file.close()

    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)
    if args.save_stan:
        try:
            print("\n\nSaving compiled model in directory %s" % log_dir)
            save_obj(sm, 'sm', log_dir)
        except:
            # Print error message
            print("could not save the stan model")
else:
    print("\n\nReading existing compiled model from directory %s" % log_dir)
    sm = load_obj('sm', log_dir)

    if args.stan_model == 1:
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
    elif args.stan_model == 2:
        param_names = ['alpha', 'yy',  'beta',
                       'Marg_cov', 'Omega_cov', 'Phi_cov']
    elif args.stan_model == 3:  # alt param of model2
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
                       'Omega_cov', 'Phi_cov']
    elif args.stan_model == 4:
        param_names = ['beta', 'alpha', 'zz', 'yy']
    elif args.stan_model == 5:
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']
    elif args.stan_model == 6:  # EFA no u's
        param_names = ['beta', 'alpha', 'zz', 'yy']
    elif args.stan_model == 7:  # EFA with u's
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov', 'Omega_cov']
    elif args.stan_model == 8:  # alt param of model 2 with cross loading on first variable no u's
        param_names = ['alpha', 'yy',  'beta', 'Phi_cov']    
    elif args.stan_model == 9:  # alt param of model 2 with cross loading on first variable
        param_names = ['alpha', 'yy',  'beta', 'Marg_cov',
                       'Omega_cov', 'Phi_cov']              
    else:
        print('model is 1:9')

############################################################
################ Fit Model ##########

if args.ppp_cv == 'ppp':  # run PPP
    print("\n\nFitting model.... \n\n")

    fit_run = sm.sampling(data=stan_data,
                          iter=args.num_samples + args.num_warmup,
                          warmup=args.num_warmup, chains=args.num_chains, n_jobs=4,
                          control={'max_treedepth': 15, 'adapt_delta': 0.99}, init=0)

    if args.save_stan:
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
                                           warmup=args.num_warmup, chains=args.num_chains, n_jobs=args.num_chains,
                                           control={'max_treedepth': 15, 'adapt_delta': 0.99}, init=0)
        if args.save_stan:
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
