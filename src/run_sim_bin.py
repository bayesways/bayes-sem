import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os

from codebase.file_utils import save_obj, load_obj
from codebase.data import gen_data_binary, gen_data_binary_1factor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_warmup", help="number of warm up iterations", type=int, default=1000)
parser.add_argument("num_samples", help="number of post-warm up iterations", type=int, default=1000)
parser.add_argument("sim_case", help="simulation case number", type=int, default=0)
parser.add_argument("stan_model", help="0:full model, 1:no u's, 2: no u's no approx zero betas ", type=int, default=0)
# Optional arguments
parser.add_argument("-num_chains","--num_chains", help="number of MCMC chains", type=int, default=1)
parser.add_argument("-seed","--random_seed", help="random seed for data generation", type=int, default=None)
parser.add_argument("-datm","--data_method", help="random seed for data generation", type=int, default=1)
parser.add_argument("-nd","--nsim_data", help="data size", type=int, default=1000)
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
    log_dir =  "./log/"+nowstr+"%s_sim%s_d%s_m%s/"%(args.task_handle,
        args.sim_case,
        args.data_method,
        args.stan_model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir+ "/"

if args.data_method == 1 or args.data_method == 3:
    model_type = 'logit'
elif args.data_method == 2 or args.data_method == 4:
    model_type = 'probit'
else:
    print("data method must be in [1,2,3,4]")


############################################################
################ Create Data or Load ##########
if args.existing_directory is None:

    print("\n\nGenerating Continuous data for case %s"%args.sim_case)
    if args.sim_case == 0 :
        data = gen_data_binary(args.nsim_data,
            method = args.data_method,
            random_seed = args.random_seed)
    elif args.sim_case == 1 :
        data = gen_data_binary_1factor(args.nsim_data,
            random_seed = args.random_seed)
    elif args.sim_case == 2 :
        data = gen_data_binary(args.nsim_data,
            cross_loadings = True, cross_loadings_level = 1,
            method = args.data_method,
            random_seed = args.random_seed)
    elif args.sim_case == 3 :
        data = gen_data_binary(args.nsim_data,
            off_diag_residual = True,
            method = args.data_method,
            random_seed = args.random_seed)
    elif args.sim_case == 4 :
        data = gen_data_binary(args.nsim_data,
            cross_loadings = True, cross_loadings_level = 2,
            method = args.data_method,
            random_seed = args.random_seed)
    elif args.sim_case == 5 :
        data = gen_data_binary(args.nsim_data,
            cross_loadings = True, cross_loadings_level = 0,
            method = args.data_method,
            random_seed = args.random_seed)
    else:
        print("Choose simulation case 0:Clean data ")
        print("Choose simulation case 1:Clean data 1 factor")
        print("Choose simulation case 2:Cross loadings")
        print("Choose simulation case 3:Off-diag residuals")

    print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))

    stan_data = dict(N = data['N'], K = 2, J = data['J'],
        DD = data['D'])
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
        #no u's, exact zeros
        with open('./codebase/stan_code/discr/CFA/%s/t_model1.stan' % model_type, 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
    elif args.stan_model == 1 :
        #no u's, exact zeros
        with open('./codebase/stan_code/discr/CFA/%s/model1.stan' % model_type, 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
    elif args.stan_model == 2 :
        #with u's of identity covariance and approx zeros
        with open('./codebase/stan_code/discr/CFA/%s/model2_prm2.stan' % model_type, 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'uu' , 'Phi_cov', 'yy']
    elif args.stan_model == 3 :
        #w u's (full covariance), approx zeros
        with open('./codebase/stan_code/discr/CFA/%s/model3_prm2.stan' % model_type, 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz' ,'uu' , 'Omega_cov', 'Phi_cov', 'yy']
    elif args.stan_model == 4 :
        #with u's (of identity covariance), exact zeros
        with open('./codebase/stan_code/discr/CFA/%s/model4.stan' % model_type, 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'uu' , 'Phi_cov', 'yy']
    elif args.stan_model == 5 :
        #no u's, exact zeros
        with open('./codebase/stan_code/discr/CFA/%s/model1_prm2.stan' % model_type, 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov',  'yy']
    elif args.stan_model == 6 :
        #no u's, exact zeros
        with open('./codebase/stan_code/discr/CFA/%s/model2_prm3.stan' % model_type, 'r') as file:
            model_code = file.read()
        param_names = ['beta', 'alpha', 'zz', 'uu' , 'Phi_cov', 'yy']
    else:
        print("Choose from 1:4}")


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
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
    elif args.stan_model == 1 :
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov', 'yy']
    elif args.stan_model == 2 :
        param_names = ['beta', 'alpha', 'zz', 'uu' , 'Phi_cov', 'yy']
    elif args.stan_model == 3 :
        param_names = ['beta', 'alpha', 'zz' , 'Phi_cov', 'yy']
    elif args.stan_model == 4 :
        param_names = ['beta', 'alpha', 'zz', 'uu' , 'Phi_cov', 'yy']
    elif args.stan_model == 5 :
        param_names = ['beta', 'alpha', 'zz', 'Phi_cov',  'yy']
    elif args.stan_model == 6 :
        param_names = ['beta', 'alpha', 'zz', 'uu' , 'Phi_cov', 'yy']
    else:
        print("Choose from 1:4}")

############################################################
################ Fit Model ##########
print("\n\nFitting model.... \n\n")

init_values = dict()

def set_initial_values(params):
    global init_values    # Needed to modify global copy of globvar
    init_values = params

set_initial_values(dict(
    beta_pos = np.ones(stan_data['K']),
    beta_free = np.ones((stan_data['K'],stan_data['K']))
    ))

def initf1():
    return init_values

fit_run = sm.sampling(data=stan_data,
    iter=args.num_samples + args.num_warmup,
    warmup=args.num_warmup, chains=args.num_chains,
    init=initf1)

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
