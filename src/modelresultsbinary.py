import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli
import datetime
import sys
import os
import itertools

from tqdm import tqdm
from codebase.file_utils import save_obj, load_obj
from scipy.special import expit
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("logdir", help="path to files", type=str, default=None)
#
# # Optional arguments
# parser.add_argument("-prm", "--print_model", help="print model on screen", type=int, default=1)
#
# args = parser.parse_args()


def to_str_pattern(y0):
    if np.ndim(y0) == 1:
        return ''.join(y0.astype(str))
    if np.ndim(y0) == 2:
        y = pd.DataFrame(y0)
        yresp = y.apply(lambda x: ''.join(x.astype(str)), axis=1)
        return yresp


def to_nparray_data(yresp):
    if type(yresp) == str:
        return np.array(list(yresp)).astype(int)
    else:
        J = len(yresp[0])
        N = yresp.shape[0]
        res = np.empty((N,J))
        for i in range(N):
            res[i] = np.array(list(yresp[i])).astype(int)
        return res


def all_possible_patterns(n):
    lst = list(map(list, itertools.product([0, 1], repeat=n)))
    return to_str_pattern(lst)


def get_avg_probs(data, ps, m):
    ## compute the pi's for the the m-th posterior sample
    N = data['N']
    L = 1000
    z_mc = multivariate_normal.rvs(np.zeros(data['K']), ps['Phi_cov'][m], size = L);
    ystr = np.empty((L, data['J']))
    for l in range(L):
        ystr[l] = ps['alpha'][m] + z_mc[l] @ ps['beta'][m].T
    pistr = expit(ystr)
    piavg = np.mean(pistr,0)

    return piavg


def get_prob_pred_data(data, ps, m):
    N = data['N']
    L = 1000
    pistr = np.empty((N, data['J']))

    for subj_i in range(N):
        z_mc = multivariate_normal.rvs(np.zeros(data['K']), ps['Phi_cov'][m], size = L);
        ystr = np.empty((L,data['J']))
        for l in range(L):
            ystr[l] = ps['alpha'][m] + z_mc[l] @ ps['beta'][m].T
        pistr[subj_i] =  np.mean(expit(ystr),0)
    return bernoulli.rvs(pistr)



def get_Ds(data, piavg):

    data_ptrn = to_str_pattern(data)
    distinct_patterns = np.unique(data_ptrn)

    N = data.shape[0]
    ## compute E_y(theta) for a specific pattern y
    Ey = dict()
    for ptrn in distinct_patterns:
        Ey[ptrn] = N * np.prod(bernoulli.pmf(k=to_nparray_data(ptrn), p = piavg));

    # compute observed pattern occurences
    Oy = dict()
    for ptrn in distinct_patterns:
        Oy[ptrn] = np.count_nonzero(data_ptrn == ptrn)

    # compute the discrepancy D
    Dy = dict()
    for ptrn in distinct_patterns:
        Dy[ptrn] = Oy[ptrn] * np.log(Oy[ptrn]/Ey[ptrn])

    return Ey, Oy, Dy
# sum(Dy.values())
