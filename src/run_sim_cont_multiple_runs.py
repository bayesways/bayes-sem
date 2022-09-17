import numpy as np
import pandas as pd
import pystan
import datetime
import os
from codebase.file_utils import save_obj, load_obj
from codebase.data import gen_data
from codebase.model_fit_cont import get_PPP
from codebase.post_process import remove_cn_dimension
import argparse

parser = argparse.ArgumentParser()
# Optional arguments
parser.add_argument(
    "-m",
    "--stan_model",
    help="0:full model, 1:no u's, 2: no u's no approx zero betas ",
    type=int,
    default=1,
)
parser.add_argument(
    "-num_warmup",
    "--num_warmup",
    help="number of warm up iterations",
    type=int,
    default=100,
)
parser.add_argument(
    "-num_samples",
    "--num_samples",
    help="number of post-warm up iterations",
    type=int,
    default=100,
)
parser.add_argument(
    "-num_chains", "--num_chains", help="number of MCMC chains", type=int, default=1
)
parser.add_argument("-nd", "--nsim_data", help="data size", type=int, default=10)
parser.add_argument(
    "-th", "--task_handle", help="hande for task", type=str, default="_"
)
parser.add_argument(
    "-xdir",
    "--existing_directory",
    help="refit compiled model in existing directory",
    type=str,
    default=None,
)
parser.add_argument(
    "-sqz",
    "--squeeze_ps",
    help="squeeze posterior samples vectors",
    type=int,
    default=0,
)
parser.add_argument("-cm", "--compile_model",
                    help="load model", type=int, default=0)
parser.add_argument("-prm", "--print_model",
                    help="print model on screen", type=int, default=0)
args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')  # ISO 8601 format
    log_dir = "./log/"+nowstr+"%s_m%s/" % (args.task_handle,
                                              args.stan_model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir + "/"

############################################################
################ Compile Model or Load ##########
path_to_stan = './codebase/stan_code/cont/'

print("\n\nReading Stan Code from model %d" % args.stan_model)
if args.stan_model == 0 :
    with open(path_to_stan+'EFA/model0.stan', 'r') as file:
        model_code = file.read()        
    param_names = ['Marg_cov', 'alpha']
elif args.stan_model == 1 :
    with open(path_to_stan+'CFA/model1.stan', 'r') as file:
        model_code = file.read()
    param_names = ['Marg_cov', 'beta', 'Phi_cov', 'sigma', 'alpha', 'Theta']
elif args.stan_model == 2 :
    with open(path_to_stan+'CFA/model2.stan', 'r') as file:
        model_code = file.read()
    param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma', 'alpha',
        'Theta', 'Omega']
elif args.stan_model == 3 :
    with open(path_to_stan+'CFA/model3.stan', 'r') as file:
        model_code = file.read()
    param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma', 'alpha',
        'Theta']
elif args.stan_model == 4 :
    with open(path_to_stan+'EFA/model1.stan', 'r') as file:
        model_code = file.read()
    param_names = ['Marg_cov', 'beta', 'sigma', 'alpha', 'Theta']
elif args.stan_model == 5 :
    with open(path_to_stan+'EFA/model2.stan', 'r') as file:
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

if args.compile_model==0:
    with open('log/compiled_models/cont/model%s/model.txt' % args.stan_model, 'r') as file:
        saved_model = file.read()
    if saved_model == model_code:
        sm = load_obj('sm', 'log/compiled_models/cont/model%s/' % args.stan_model)
        if args.stan_model == 0:
                param_names = ['Marg_cov', 'alpha']
        elif args.stan_model == 1:
            param_names = ['Marg_cov', 'beta', 'Phi_cov', 'sigma', 'alpha', 'Theta']
        elif args.stan_model == 2:
            param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma', 'alpha',
                'Theta', 'Omega']
        elif args.stan_model == 3:
            param_names = ['Marg_cov',  'beta', 'Phi_cov', 'sigma', 'alpha',
                'Theta']
        elif args.stan_model == 4:
            param_names = ['Marg_cov', 'beta', 'sigma', 'alpha', 'Theta']
        elif args.stan_model == 5:
            param_names = ['Marg_cov',  'beta', 'sigma', 'alpha', 'Theta', 'Omega']
        else:
            print("model option should be in [0,1,2,3]")

else:
    print("\n\nCompiling model")
    sm = pystan.StanModel(model_code=model_code, verbose=False)
    try:
        print("\n\nSaving compiled model in directory %s" % log_dir)
        save_obj(sm, 'sm', 'log/compiled_models/cont/model%s/' % args.stan_model)
        file = open('log/compiled_models/cont/model%s/model.txt' %
                    args.stan_model, "w")
        file.write(model_code)
        file.close()
    except:
        print("Couldn't save model in model bank")

print("\n\nSaving compiled model in directory %s" % log_dir)
save_obj(sm, 'sm', log_dir)



############################################################
################ Create Data or Load ##########

for random_seed in range(20):
    data = gen_data(args.nsim_data,
                        off_diag_residual=False,
                        random_seed = random_seed)

    print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))
    data['sigma_prior'] = np.diag(np.linalg.inv(np.cov(data['y'], rowvar=False)))
    stan_data = dict(N = data['N'], K = data['K'], J = data['J'],
        yy = data['y'], sigma_prior = data['sigma_prior'])
    print("\n\nSaving data to directory %s"% log_dir)
    save_obj(stan_data, 'stan_data'+str(random_seed), log_dir)
    save_obj(data, 'data'+str(random_seed), log_dir)


    ############################################################
    ################ Fit Model ##########

    fit_run = sm.sampling(
        data=stan_data,
        iter=args.num_samples + args.num_warmup,
        warmup=args.num_warmup, chains=args.num_chains,
        n_jobs=4,
        control={'max_treedepth': 10, 'adapt_delta': 0.9},
        init = 0
        )
   
    try:
        print("\n\nSaving fitted model in directory %s" % log_dir)
        save_obj(fit_run, 'fit'+str(random_seed), log_dir)
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
    save_obj(ps, 'ps'+str(random_seed), log_dir)

    num_chains = ps['alpha'].shape[1]
    num_samples = ps['alpha'].shape[0]

    for name in ['alpha', 'Marg_cov']:
        ps[name] = remove_cn_dimension(ps[name])

    PPP_vals = get_PPP(stan_data, ps, args.nsim_ppp)
    save_obj(PPP_vals, 'PPP_vals'+str(random_seed), log_dir)

    ppp = np.sum(PPP_vals[:, 0] < PPP_vals[:, 1])/args.nsim_ppp
    save_obj(ppp, 'ppp'+str(random_seed), log_dir)

