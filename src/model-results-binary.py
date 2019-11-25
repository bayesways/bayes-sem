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


data = load_obj("stan_data", log_dir)
ps = load_obj('ps', log_dir)

num_samples = ps['alpha'].shape[0]


post_yy = np.empty((num_samples,data['N'],data['J']))
for i in range(num_samples):
    if 'uu' in ps.keys():
        post_yy[i] = ps['alpha'][i] + np.outer(ps['zz'][i], ps['beta'][i]) + ps['uu'][i]
    else:
        post_yy[i] = ps['alpha'][i] + np.outer(ps['zz'][i], ps['beta'][i])
ps['yy'] = post_yy
ps['pp'] = expit(post_yy)


def f_score(data, score):
    assert score>=1
    J = data.shape[1]
    df = pd.DataFrame(data)
    df.columns = ['x'+str(i) for i in range(1,J+1)]
    df.insert(column='s', value=np.sum(data,1), loc=0)
    res = df[(df['s'] == score)].shape[0]
    return res


def f_item_score(data, item, score):
    J = data.shape[1]
    assert score>=1
    assert item>=1
    assert item<=J
    df = pd.DataFrame(data)
    df.columns = ['x'+str(i) for i in range(1,data.shape[1]+1)]
    df.insert(column='s', value=np.sum(data,1), loc=0)
    res = df[(df['s'] == score) & df['x'+str(item)] == 1].shape[0]
    return res


def prob_items(probs, data, score):
    assert score>=1
    probdf = pd.DataFrame(probs)
    probdf.insert(column='s', value=np.sum(data,1), loc=0)
    res = np.mean(probdf[probdf.s == score].values[:,1:], 0)
    return res

def compute_Q(probs, data):
    J = data.shape[1]
    fjs = np.empty((J, J))
    pjs = np.empty((J, J))
    fs = np.empty(J)
    for s in range(1,J+1):
        fs[s-1] = f_score(data, s)
        pjs[:,s-1] = prob_items(probs, data, s)
        for j in range(1,J+1):
            fjs[j-1,s-1] = f_item_score(data, j, s)

    # 2.3.3 Yen's Index (Q1) - p.37 of thesis
    # res = np.empty((J,J))
    # for s in range(J):
    #     for j in range(J):
    #         res[j,s] = (fjs[j,s] - fs[s]*pjs[j,s])**2/(fs[s]*pjs[j,s]*(1 - pjs[j,s]))

    # 2.3.5 McKingley and Mills Index (G2) - p.38 of thesis
    res = np.empty((J,J))
    for s in range(J):
        for j in range(J):
            # res[j,s] = fjs[j,s] * (np.log(fjs[j,s])- np.log(fs[s]*pjs[j,s]))+\
            #         (1 - fjs[j,s]) * np.log((1 - fjs[j,s])/(1 - fs[s]*pjs[j,s]))

            # numerical more stable than above
            res[j,s] = fjs[j,s] * (np.log(fjs[j,s])- np.log(fs[s]*pjs[j,s]))+\
                   (1 - fjs[j,s]) * (np.log(fjs[j,s]-1 + 1e-10) - np.log(fs[s]*pjs[j,s]-1 + 1e-10))
    return res


def compute_all_Qs(probs, subsample_n):
    num_samples = probs[0].shape[0]
    J = probs[0].shape[1]
    Ds = np.empty((num_samples,J,2))
    for i in tqdm(range(num_samples)[::subsample_n]):
        Ds[i,:,0] = np.sum(compute_Q(probs[i], data['DD']),0)
        data_pred = bernoulli.rvs(probs[i])
        Ds[i,:,1] = np.sum(compute_Q(probs[i], data_pred),0)
        np.sum(Ds[:,:,0] < Ds[:,:,1], 0) / num_samples
    return np.sum(Ds[:,:,0] < Ds[:,:,1], 0) / num_samples

res = compute_all_Qs(ps['pp'], args.subsample)
print(res)
