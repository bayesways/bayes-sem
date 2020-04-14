import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli, norm
import datetime
import sys
import os
import itertools

from tqdm.notebook import tqdm
from codebase.file_utils import save_obj, load_obj
from scipy.special import expit
from scipy.special import ndtri
import argparse


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


def get_all_possible_patterns(n):
    lst = list(map(list, itertools.product([0, 1], repeat=n)))
    return to_str_pattern(lst)


def get_exp_probs(data, ps, m , L=100):
    ## compute the pi's for the the m-th posterior sample
    if 'zz' in ps.keys():
        z_mc = multivariate_normal.rvs(np.zeros(data['K']),
            ps['Phi_cov'][m], size = L)
        ystr = np.empty((L, data['J']))
        for l in range(L):
            ystr[l] = ps['alpha'][m] + z_mc[l] @ ps['beta'][m].T
    elif 'Marg_cov' in ps.keys():
        ystr = multivariate_normal.rvs(ps['alpha'][m],
            ps['Marg_cov'][m], size = L)
    else:
        print("No matching model")

    # logit
    pistr = expit(ystr)

    # probit
    # pistr = norm.cdf(ystr)

    return pistr


def get_exp_probs2(data, ps, m , L=100):
    num_chains = 4
    ystr = np.empty((num_chains,L, data['J']))
    for cn in range(num_chains):
        if 'zz' in ps.keys():
            z_mc = multivariate_normal.rvs(np.zeros(data['K']),
                ps['Phi_cov'][m, cn], size = L)
            for l in range(L):
                ystr[cn, l] = ps['alpha'][m, cn] + z_mc[l] @ ps['beta'][m, cn].T
        elif 'Marg_cov' in ps.keys():
            ystr[cn] = multivariate_normal.rvs(ps['alpha'][m, cn],
                ps['Marg_cov'][m, cn], size = L)
        else:
            print("No matching model")

    # logit
    pistr = np.mean(expit(ystr),0)

    # probit
    # pistr = norm.cdf(ystr)

    return pistr


def get_probs(data, ps, m, cn):
    pistr = expit(ps['yy'][m, cn])
    return pistr


def get_Ey(data_ptrn, prob, N):
    distinct_patterns = np.unique(data_ptrn)
    ## compute E_y(theta) for a specific pattern y
    Ey = dict()
    for ptrn in distinct_patterns:
        prob_matrix = bernoulli.logpmf(k=to_nparray_data(ptrn), p = prob)
        Ey[ptrn] = N * np.mean(np.exp(np.sum(prob_matrix,1)),0)
    return Ey


def get_Oy(data_ptrn):
    distinct_patterns = np.unique(data_ptrn)
    # compute observed pattern occurences
    Oy = dict()
    for ptrn in distinct_patterns:
        Oy[ptrn] = np.count_nonzero(data_ptrn == ptrn)
    return Oy


def get_Dy(Oy, Ey, data_ptrn):
    distinct_patterns = np.unique(data_ptrn)
    # compute the discrepancy D
    Dy = dict()
    for ptrn in distinct_patterns:
        Dy[ptrn] = Oy[ptrn] * np.log(Oy[ptrn]/Ey[ptrn])

    return Dy


def get_PPP(data, ps, cn, nsim = 100):

    nsim_N = ps['alpha'].shape[0]
    skip_step = int(nsim_N/nsim)

    data_ptrn = to_str_pattern(data['D'])
    Oy = get_Oy(data_ptrn)

    PPP_vals = np.empty((nsim, 2))
    for m_ind in tqdm(range(nsim)):
        m = skip_step*m_ind
        # compute Dy
        pi = get_probs(data, ps, m, cn)
        Ey = get_Ey(data_ptrn, pi, data['N'])
        Dy = get_Dy(Oy, Ey, data_ptrn)

        # compute Dy
        ppdata = bernoulli.rvs(get_probs(data, ps, m, cn))
        ppddata_ptrn = to_str_pattern(ppdata)

        Oystr = get_Oy(ppddata_ptrn)
        Eystr = get_Ey(ppddata_ptrn, pi, data['N'])
        Dystr = get_Dy(Oystr, Eystr, ppddata_ptrn)

        PPP_vals[m_ind,0] = sum(Dy.values())
        PPP_vals[m_ind,1] = sum(Dystr.values())

    return PPP_vals, Dy, Dystr
