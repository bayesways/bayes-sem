import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli
from tqdm import tqdm
from codebase.file_utils import save_obj, load_obj
from scipy.special import expit

def to_str_pattern(y0):
    if np.ndim(y0) == 1:
        return "".join(y0.astype(str))
    if np.ndim(y0) == 2:
        y = pd.DataFrame(y0)
        yresp = y.apply(lambda x: "".join(x.astype(str)), axis=1)
        return yresp


def to_nparray_data(yresp):
    if type(yresp) == str:
        return np.array(list(yresp)).astype(int)
    else:
        J = len(yresp[0])
        N = yresp.shape[0]
        res = np.empty((N, J))
        for i in range(N):
            res[i] = np.array(list(yresp[i])).astype(int)
        return res


def get_probs(data, ps, m, cn):
    pistr = expit(ps["yy"][m, cn])
    return pistr


def get_probs(data, ps, m, cn):
    pistr = expit(ps["yy"][m, cn])
    return pistr


def get_Ey(data_ptrn, prob, N):
    distinct_patterns = np.unique(data_ptrn)
    ## compute E_y(theta) for a specific pattern y
    Ey = dict()
    for ptrn in distinct_patterns:
        prob_matrix = bernoulli.logpmf(k=to_nparray_data(ptrn), p=prob)
        Ey[ptrn] = N * np.mean(np.exp(np.sum(prob_matrix, 1)), 0)
    return Ey


# def get_Ey(data_ptrn, prob_logitinv, N):
#     distinct_patterns = np.unique(data_ptrn)
#     ## compute E_y(theta) for a specific pattern y
#     Ey = dict()
#     for ptrn in distinct_patterns:
#         prob_vector = bernoulli.logpmf(k=to_nparray_data(ptrn), p=expit(prob_logitinv))
#         Ey[ptrn] = N * np.exp(np.sum(prob_vector))
#         # set_trace()
#     return Ey


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
        Dy[ptrn] = Oy[ptrn] * np.log(Oy[ptrn] / Ey[ptrn])

    return Dy


def get_PPP(data, ps, cn, nsim=100):

    nsim_N = ps["alpha"].shape[0]
    skip_step = int(nsim_N / nsim)

    data_ptrn = to_str_pattern(data["D"])
    Oy = get_Oy(data_ptrn)

    PPP_vals = np.empty((nsim, 2))
    for m_ind in tqdm(range(nsim)):
        m = skip_step * m_ind
        # compute Dy
        pi = get_probs(data, ps, m, cn)
        Ey = get_Ey(data_ptrn, pi, data["N"])
        Dy = get_Dy(Oy, Ey, data_ptrn)

        # compute Dy
        ppdata = bernoulli.rvs(get_probs(data, ps, m, cn))
        ppddata_ptrn = to_str_pattern(ppdata)

        Oystr = get_Oy(ppddata_ptrn)
        Eystr = get_Ey(ppddata_ptrn, pi, data["N"])
        Dystr = get_Dy(Oystr, Eystr, ppddata_ptrn)

        PPP_vals[m_ind, 0] = sum(Dy.values())
        PPP_vals[m_ind, 1] = sum(Dystr.values())

    return PPP_vals


# def get_lgscr(ps, data, nsim):
#     mcmc_length = ps['alpha'].shape[0]*ps['alpha'].shape[1]
#     num_chains = ps['alpha'].shape[1]

#     if nsim>mcmc_length:
#         print('nsim > posterior sample size')
#         print('Using nsim = %d'%mcmc_length)
#         nsim = mcmc_length
#     skip_step = int(mcmc_length/nsim)
#     post_y_invlogit = np.vstack(
#         np.squeeze(
#             np.split(ps['yy'],num_chains,  axis=1))
#             )
#     # set_trace()

#     skip_step = int(mcmc_length/nsim)
#     data_ptrn = to_str_pattern(data['test']['DD'])
#     Oy = get_Oy(data_ptrn)

#     # method 1
#     # logscore with values fixed at posterior mean
#     m_alpha = alphas.mean(axis=0)
#     m_Cov = covs.mean(axis=0)
#     post_y = multivariate_normal.rvs(mean=m_alpha, cov = m_Cov, size=10000)


#     # for m_ind in tqdm(range(nsim)):
#     #     m = skip_step*m_ind
#     #     pi = get_probs(data, ps, m, cn)
#     #     pi_avg = pi.mean(axis=0)
#     #     Ey = get_Ey(data_ptrn, pi_avg, data['test']['N'])
#     #     Dy = get_Dy(Oy, Ey, data_ptrn)
#     #     lgscr_vals[m_ind, cn] = sum(Dy.values())

#     return scores


def get_lgscr(ps, data, nsim, method_num):

    mcmc_length = ps["alpha"].shape[0] * ps["alpha"].shape[1]
    num_chains = ps["alpha"].shape[1]
    dim_K = ps["beta"].shape[-1]
    dim_J = ps['alpha'].shape[2]

    if nsim is None:
        print("Using all %d posterior samples"%mcmc_length)
        nsim = mcmc_length
    else:    
        if nsim > mcmc_length:
            print("nsim > posterior sample size")
            print("Using nsim = %d" % mcmc_length)
            nsim = mcmc_length
    skip_step = int(mcmc_length / nsim)
    stacked_ps = dict()
    for name in ps.keys():
        stacked_ps[name] = np.vstack(np.squeeze(np.split(ps[name], num_chains, axis=1)))

    data_ptrn = to_str_pattern(data["test"]["DD"])
    Oy = get_Oy(data_ptrn)

    if method_num == 1:
        # method 1
        m_alpha = stacked_ps["alpha"].mean(axis=0)
        if "Marg_cov" in stacked_ps.keys():
            m_Marg_cov = stacked_ps["Marg_cov"].mean(axis=0)
            post_y = multivariate_normal.rvs(mean=m_alpha, cov=m_Marg_cov, size=nsim)
        else:
            m_beta = stacked_ps["beta"].mean(axis=0)
            if "Phi_cov" in stacked_ps.keys():
                m_Phi_cov = stacked_ps["Phi_cov"].mean(axis=0)
            else:
                m_Phi_cov = np.eye(dim_K)
            zz_from_prior = multivariate_normal.rvs(
                mean=np.zeros(dim_K), cov=m_Phi_cov, size=nsim
            )
            post_y = m_alpha + zz_from_prior @ m_beta.T
   
    elif method_num == 2:
        # method 2
        post_y = np.empty((nsim, dim_J))
        for m_ind in tqdm(range(nsim)):
            m = skip_step*m_ind
            m_alpha = stacked_ps["alpha"][m]
            if "Marg_cov" in stacked_ps.keys():
                m_Marg_cov = stacked_ps["Marg_cov"][m]
                post_y_sample = multivariate_normal.rvs(mean=m_alpha, cov=m_Marg_cov, size=1)
            else:
                m_beta = stacked_ps["beta"][m]
                if "Phi_cov" in stacked_ps.keys():
                    m_Phi_cov = stacked_ps["Phi_cov"][m]
                else:
                    m_Phi_cov = np.eye(dim_K)
                zz_from_prior = multivariate_normal.rvs(
                    mean=np.zeros(dim_K), cov=m_Phi_cov, size=1
                )
                post_y_sample = m_alpha + zz_from_prior @ m_beta.T
            post_y[m_ind] = post_y_sample
    else:
        print('method_num not found')
        
    
    Ey = get_Ey(data_ptrn, expit(post_y), data["test"]["N"])
    Dy = get_Dy(Oy, Ey, data_ptrn)
    scores = sum(Dy.values())

    return scores
