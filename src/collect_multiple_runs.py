import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from codebase.file_utils import save_obj, load_obj
from codebase.post_process import remove_cn_dimension
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--log_dir", help="directory of results", type=str, default=None)

# Optional arguments


args = parser.parse_args()


def clean_samples(ps0):
    ps = ps0.copy()
    num_chains = ps['alpha'].shape[1]
    num_samples = ps['alpha'].shape[0]
    for chain_number in range(ps['alpha'].shape[1]):
        for i in range(num_samples):
            sign1 = np.sign(ps['beta'][i,chain_number,0,0])
            sign2 = np.sign(ps['beta'][i,chain_number,3,1])
            ps['beta'][i,chain_number,:3,0] = ps['beta'][i,chain_number,:3,0] * sign1
            ps['beta'][i,chain_number,3:,1] = ps['beta'][i,chain_number,3:,1] * sign2

            if 'Phi_cov' in ps.keys():
                ps['Phi_cov'][i,chain_number,0,1] = sign1 * sign2 * ps['Phi_cov'][i,chain_number,0,1]
                ps['Phi_cov'][i,chain_number,1,0] = ps['Phi_cov'][i,chain_number,0,1]
    
    return ps

def get_point_estimates(ps0, param_name, estimate_name):
    ps = remove_cn_dimension(
        clean_samples(ps0)[param_name]
    )
    if estimate_name == 'mean':
        return np.mean(ps,axis=0)
    elif estimate_name == 'median':
        return np.median(ps, axis=0)

log_dir = args.log_dir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"


nsim = 80

for i in range(nsim):
    ps =  clean_samples(load_obj('ps'+str(i), log_dir))
    quant = np.quantile(
        remove_cn_dimension(ps['beta']),
        [0.025, 0.975],
        axis=0
        )
    save_obj(quant, 'q_beta'+str(i), log_dir)
    
    quant = np.quantile(
        remove_cn_dimension(ps['Phi_cov']),
        [0.025, 0.975],
        axis=0
        )
    save_obj(quant, 'q_Phi_cov'+str(i), log_dir)


param = 'beta'
estimate_name = 'mean'
estimates = np.empty((nsim, 6, 2))
for i in range(nsim):
    estimates[i] = get_point_estimates(
        load_obj('ps'+str(i), log_dir),
        param,
        estimate_name
    )
save_obj(estimates, 'beta_mean', log_dir)


estimate_name = 'median'
estimates = np.empty((nsim, 6, 2))
for i in range(nsim):
    estimates[i] = get_point_estimates(
        load_obj('ps'+str(i), log_dir),
        param,
        estimate_name
    )
save_obj(estimates, 'beta_median', log_dir)


param = 'Phi_cov'
estimate_name = 'mean'
estimates = np.empty((nsim, 2, 2))
for i in range(nsim):
    estimates[i] = get_point_estimates(
        load_obj('ps'+str(i), log_dir),
        param,
        estimate_name
    )
save_obj(estimates, 'Phi_mean', log_dir)

estimate_name = 'median'
estimates = np.empty((nsim, 2, 2))
for i in range(nsim):
    estimates[i] = get_point_estimates(
        load_obj('ps'+str(i), log_dir),
        param,
        estimate_name
    )
save_obj(estimates, 'Phi_median', log_dir)

param = 'alpha'
estimate_name = 'mean'
estimates = np.empty((nsim, 1, 6))
for i in range(nsim):
    estimates[i] = get_point_estimates(
        load_obj('ps'+str(i), log_dir),
        param,
        estimate_name
    )
save_obj(estimates, 'alpha_mean', log_dir)

estimate_name = 'median'
estimates = np.empty((nsim, 1, 6))
for i in range(nsim):
    estimates[i] = get_point_estimates(
        load_obj('ps'+str(i), log_dir),
        param,
        estimate_name
    )
save_obj(estimates, 'alpha_median', log_dir)