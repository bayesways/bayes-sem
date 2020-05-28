import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, bernoulli
from scipy.special import expit
import datetime
import sys
import os
from codebase.file_utils import save_obj, load_obj
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=1)
parser.add_argument("-sbsm", "--subsample", help="subsample every nth sample", type=int, default=10)

args = parser.parse_args()

print("\n\nPrinting Stan model code \n\n")


log_dir = args.logdir
if log_dir[-1] != "/":
    print("\n\nAppending `/`-character at the end of directory")
    log_dir = log_dir+ "/"

if bool(args.print_model):
    file = open(log_dir+"model.txt", "r")
    print(file.read())
    file.close()


data = load_obj("data", log_dir)
ps = load_obj('ps', log_dir)

num_samples = ps['alpha'].shape[0]

def order_data(data, zz, beta):
    load_sign = np.sign(np.mean(beta))
    J = data.shape[1]
    df = pd.DataFrame(data)
    df.columns = ['x'+str(i) for i in range(1,J+1)]
    df.insert(column='z', value=zz*load_sign, loc=0)
    return df.sort_values('z')


def exp_k(df, k, alpha, beta, num_gprs=10):
    load_sign = np.sign(np.mean(beta))
    N = df.shape[0]/num_gprs
    z_bar = np.mean(np.array_split(df,num_gprs)[k]['z'])
    y_bar = alpha + z_bar * beta * load_sign
    return np.round(expit(y_bar)*N,0)


def obs_k(df, k, num_gprs=10):
    return np.array_split(df,num_gprs)[k].drop('z',1).sum(0).values


def obs_star_k(df, k, alpha, beta, num_gprs=10):
    load_sign = np.sign(np.mean(beta))
    y_star = alpha +  np.outer(df.z, beta*load_sign)
    data_pred = pd.DataFrame(bernoulli.rvs(expit(y_star)))
    return np.array_split(data_pred,num_gprs)[k].sum(0).values


def G2(N, E, O):
    if (O != 0).all():
        return N*(O*np.log(O/E) + (1-O) * np.log((1-O)/(1-E)))
    else:
        return np.nan

def G2_item(N, E, O, item):
#     assert O[item] != 0
    return N*(O[item]*np.log(O[item]/E[item]) + (1-O[item]) * np.log((1-O[item])/(1-E[item])))


GG2 = np.empty((num_samples,10,6))
GG2_post = np.empty((num_samples,10,6))
N = 100
for i in tqdm(range(1000)):
    for k in range(10):
        df = order_data(data['D'], ps['zz'][i], ps['beta'][i])
        O = obs_k(df, k)
        Ostar = obs_star_k(df, k, ps['alpha'][i], ps['beta'][i])
        E = exp_k(df, k, ps['alpha'][i], ps['beta'][i])
        for item in range(6):
            if O[item] ==0:
                GG2[i,k,item] = np.nan
            else:
                GG2[i,k,item] = G2_item(N, E, O, item)
            if Ostar[item] ==0:
                GG2_post[i,k,item] = np.nan
            else:
                GG2_post[i,k,item] = G2_item(N, E, Ostar, item)

Des = np.empty(6)
for item in range(6):
    Des[item] = np.sum(pd.DataFrame(GG2[:,:,item]).sum(1) < pd.DataFrame(GG2_post[:,:,item]).sum(1))

print(np.round((Des/num_samples)*100,0))
