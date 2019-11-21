import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os

from codebase.file_utils import save_obj, load_obj
from codebase.data import gen_data, gen_data_binary
import argparse


############################################################
###### MCMC Auxiliary Functions ############################


def mcmc(stan_data, init_vals, stepsize, inv_metric, control, nsim):
    sm = load_obj('sm', log_dir)
    set_initial_values(params)

    fit_run = sm.sampling(data=stan_data,
                      warmup=0, iter=args.num_samples, chains=1,
                      init=init_vals, control=control, check_hmc_diagnostics=False)

    print("\n\nSaving posterior samples in %s"%log_dir)
    stan_samples= fit_run.extract(permuted=False, pars=param_names, inc_warmup = False )  # return a dictionary of arrays

    if args.num_chains ==1:
        ps = dict()
        for name in param_names:
            ps[name] = np.squeeze(stan_samples[name])
    else:
        ps = stan_samples

    output_init_values = fit_run.get_last_position()

    return ps, output_init_values

############################################################
################


parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("sim_case", help="simulation case number", type=int, default=1)
parser.add_argument("stan_model", help="1:no u's, 2: no u's no approx zero betas ", type=int, default=0)
# Optional arguments
parser.add_argument("-sim_case", "--sim_case", help="simulation case number", type=int, default=0)
parser.add_argument("-num_chains","--num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("-seed","--random_seed", help="random seed for data generation", type=int, default=None)
parser.add_argument("-nd","--nsim_data", help="data size", type=int, default=500)
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
    print("\n\nReading data case %s"%args.sim_case)
    if args.sim_case == 0 :
        print("\n\nReading data for case %s"%args.sim_case)
        df = pd.read_csv("../dat/LSAT.csv")
        data = dict()
        data['N'] = df.shape[0]
        data['K'] = 1
        data['J'] = df.shape[1]
        data['D'] = df.values.astype(int)
    elif args.sim_case == 1 :
        print("\n\nReading data for case %s"%args.sim_case)
        df = pd.read_csv("../dat/clean_iri_data.csv")
        data = dict()
        data['N'] = df.shape[0]
        data['K'] = 1
        data['subj_id'] = df.iloc[:,0]
        data['J'] = df.shape[1]-2
        data['D'] = df.iloc[:,2:].astype(int).values
    elif args.sim_case == 2 :
        print("\n\nReading data for case %s"%args.sim_case)
        df = pd.read_csv("../dat/clean_sample_iri_data.csv")
        data = dict()
        data['N'] = df.shape[0]
        data['K'] = 1
        data['subj_id'] = df.iloc[:,0]
        data['J'] = df.shape[1]-2
        data['D'] = df.iloc[:,2:].astype(int).values
    elif args.sim_case == 3 :
        print("\n\nReading data for case %s"%args.sim_case)
        df = pd.read_csv("../dat/WIRS.csv")
        data = dict()
        data['N'] = df.shape[0]
        data['K'] = 1
        data['J'] = df.shape[1]
        data['D'] = df.astype(int).values
    else:
        print("Choose sim case {0:LSAT data"
        "1: Irini's cheating dataset"
        "2: Irini's cheating dataset (sample)"
        "3: Irini's WIRS dataset")

    print("\n\nN=%d, J=%d, K=%d"%(data['N'],data['J'], data['K']))
    stan_data = dict(N = data['N'], K = data['K'], J = data['J'],
        DD = data['D'])
######### #### #########
    # stan_data = dict(N=10, y=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1])
######### #### #########

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
    if args.stan_model == 1 :
        #no u's
        with open('./codebase/stan_code/discr/CFA/model1_1f.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha']
    elif args.stan_model == 2 :
        #with u's and full covariance matrix of u's
        with open('./codebase/stan_code/discr/CFA/model2_1f.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'uu' ]
    elif args.stan_model == 3 :
        #with u's and identity covariance matrix
        with open('./codebase/stan_code/discr/CFA/model2_1f_2.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'uu' ]
    elif args.stan_model == 4 :
        # scalar loadings
        with open('./codebase/stan_code/discr/CFA/model2_1f_4.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'uu' ]
    elif args.stan_model == 5 :
        # positive u's
        with open('./codebase/stan_code/discr/CFA/model2_1f_5.stan', 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'uu' ]
    else:
        print("Choose from 1:5}")

    if bool(args.print_model):
        print(model_code)
    file = open(log_dir+"model.txt", "w")
    file.write(model_code)
    file.close()

    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)

    ######### TEST #########
    # model_code = """
    #     data {
    #       int<lower=0> N;
    #       int<lower=0,upper=1> y[N];
    #     }
    #     parameters {
    #       real<lower=0,upper=1> theta;
    #     }
    #     model {
    #       theta ~ beta(0.5, 0.5);  // Jeffreys' prior
    #       for (n in 1:N)
    #         y[n] ~ bernoulli(theta);
    #     }
    # """
    # sm = pystan.StanModel(model_code=model_code, verbose=False)
    ######### #### #########

    try:
        print("\n\nSaving compiled model in directory %s"%log_dir)
        save_obj(sm, 'sm', log_dir)
    except:
        # Print error message
        print("could not save the stan model")


else:
    print("\n\nReading existing compiled model from directory %s"%log_dir)
    sm = load_obj('sm', log_dir)
    if args.stan_model == 1 :
        param_names = ['beta', 'alpha']
    elif args.stan_model == 2 :
        param_names = ['beta', 'alpha', 'zz', 'uu' ]
    elif args.stan_model == 3 :
        param_names = ['beta', 'alpha', 'zz', 'uu' ]
    elif args.stan_model == 4 :
        param_names = ['beta', 'alpha', 'zz' , 'uu']
    elif args.stan_model == 5 :
        param_names = ['beta', 'alpha', 'zz', 'uu' ]
    else:
        print("Choose stan model {1:exact zeros no u's, 2: full factor model}")


############################################################
################ Fit Model ##########

# param_names = ['theta']
print("\n\nRunning warm up.... \n\n")
fit_warmup = sm.sampling(data=stan_data,
            iter=args.num_warmup, chains=args.num_chains,
            check_hmc_diagnostics=False,
            control={
                # "metric" : "diag_e",
                  "adapt_delta" : 0.99,
                  "max_treedepth" : 14,
                  "adapt_engaged" : True,
                     })

# stepsize = fit_warmup.get_stepsize()
# inv_metric = fit_warmup.get_inv_metric(as_dict=True)
# init_vals = fit_warmup.get_last_position()

stepsize = fit_warmup.get_stepsize()
inv_metric = fit_warmup.get_inv_metric(as_dict=True)
init = fit_warmup.get_last_position()

import pdb; pdb.set_trace();

stan_samples= fit_warmup.extract(permuted=False, pars=param_names)  # return a dictionary of arrays
if args.num_chains==1:
    ps = dict()
    for name in param_names:
        ps[name] = np.squeeze(stan_samples[name])
else:
    ps = stan_samples

try:
    save_obj(ps, 'warmup_ps', log_dir)
except:
    # Print error message
    print("could not save the posterior samples")


control={"stepsize":stepsize,
             "inv_metric" : inv_metric,
             # "metric" : "diag_e",
             "adapt_engaged" : False,
             "adapt_delta" : 0.99,
             "max_treedepth" : 13,
            }

# for k in range(2):
#     ps, init_vals = mcmc(stan_data, init_vals, stepsize, inv_metric, control, nsim = args.num_samples)
#     try:
#         save_obj(ps, 'ps_'+str(k), log_dir)
#     except:
#         # Print error message
#         print("could not save the posterior samples")


for k in range(2):
    fit = sm.sampling(data=stan_data,
                       warmup=0, iter=args.num_samples, chains=1,
                       control=control,
                       init=init)


    stepsize = fit.get_stepsize()
    inv_metric = fit.get_inv_metric(as_dict=True)
    init = fit.get_last_position()

    pdb.set_trace();


    control = {"stepsize" : stepsize,
               "inv_metric" : inv_metric,
               "adapt_engaged" : False
               }

    stan_samples= fit.extract(permuted=False, pars=param_names, inc_warmup = False)  # return a dictionary of arrays
    if args.num_chains==1:
        ps = dict()
        for name in param_names:
            ps[name] = np.squeeze(stan_samples[name])
    else:
        ps = stan_samples
    try:
        save_obj(ps, 'ps_'+str(k), log_dir)
    except:
        # Print error message
        print("could not save the posterior samples")
