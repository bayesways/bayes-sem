import numpy as np
import pandas as pd
import pystan
import datetime
import sys
import os
from codebase.file_utils import save_obj, load_obj
from codebase.post_process import samples_to_df, get_post_df, remove_cn_dimension
import altair as alt
alt.data_transformers.disable_max_rows()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "log_dir", help="number of warm up iterations", type=str, default=None)

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

    
def get_credible_interval_beta(ps):
    df = get_post_df(ps)
    df_quant = df.groupby(['row', 'col'])[['value']].quantile(0.025).reset_index()
    df_quant.rename({'value':'q1'}, axis=1, inplace=True)
    df_quant2 = df.groupby(['row', 'col'])[['value']].quantile(0.975).reset_index()
    df_quant2.rename({'value':'q2'}, axis=1, inplace=True)

    df = df_quant.merge(df_quant2, on=['row', 'col'])
    dd = pd.DataFrame(data['beta'], columns=['0', '1'])
    dd['row'] = np.arange(dd.shape[0])
    dd = dd.melt(id_vars='row', var_name='col', value_name = 'data')
    dd['col'] = dd.col.astype(int)

    plot_data = df.merge(dd, on=['row', 'col'])
    plot_data['index'] = 'row ' + plot_data.row.astype(str)+' .col '+plot_data.col.astype(str)

    c1 = alt.Chart(plot_data).mark_bar(opacity=0.6).encode(
            alt.X('q1', title=None),
            alt.X2('q2', title=None))
    c2 = alt.Chart(plot_data).mark_point(opacity=1, color='red').encode(
            alt.X('data', title=None)
    )
    return (c1+c2
            ).facet(
                   'index',
                columns=2
                )


def get_credible_interval_Phi(ps):
    df = get_post_df(ps)
    df = df[(df.row==0)&(df.col==1)]
    df_quant = df.groupby(['row', 'col'])[['value']].quantile(0.025).reset_index()
    df_quant.rename({'value':'q1'}, axis=1, inplace=True)
    df_quant2 = df.groupby(['row', 'col'])[['value']].quantile(0.975).reset_index()
    df_quant2.rename({'value':'q2'}, axis=1, inplace=True)
    plot_data = df_quant.merge(df_quant2, on=['row', 'col'])
    plot_data['data'] = data['Phi_cov'][0,1]
    plot_data
    plot_data['index'] = 'row ' + plot_data.row.astype(str)+' .col '+plot_data.col.astype(str)

    c1 = alt.Chart(plot_data).mark_bar(opacity=0.6).encode(
            alt.X('q1', title=None),
            alt.X2('q2', title=None))

    c2 = alt.Chart(plot_data).mark_point(opacity=1, color='red').encode(
            alt.X('data', title=None)
    )
    return (c1+c2
            ).facet(
                   'index',
                columns=2
                )


log_dir = args.log_dir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"


nsim = 20


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
