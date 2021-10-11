import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli
from tqdm import tqdm
from codebase.file_utils import save_obj, load_obj
from scipy.special import expit
from pdb import set_trace


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


def get_probs(data, ps, m):
    pistr = expit(ps["yy"][m])
    return pistr


def get_Ey(data_ptrn, prob, N):
    distinct_patterns = np.unique(data_ptrn)
    ## compute E_y(theta) for a specific pattern y
    Ey = dict()
    for ptrn in distinct_patterns:
        prob_matrix = bernoulli.logpmf(k=to_nparray_data(ptrn), p=prob)
        Ey[ptrn] = N * np.mean(np.exp(np.sum(prob_matrix, 1)), 0)
    return Ey


def get_response_probs(data_ptrn, prob):
    distinct_patterns = np.unique(data_ptrn)
    ## compute E_y(theta) for a specific pattern y
    Ey = dict()
    for ptrn in distinct_patterns:
        prob_matrix = bernoulli.logpmf(k=to_nparray_data(ptrn), p=prob)
        Ey[ptrn] = np.mean(np.exp(np.sum(prob_matrix, 1)), 0)
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
        Dy[ptrn] = Oy[ptrn] * np.log(Oy[ptrn] / Ey[ptrn])

    return Dy


def get_PPP(data, ps, nsim=100):

    nsim_N = ps["alpha"].shape[0]
    skip_step = int(nsim_N / nsim)

    data_ptrn = to_str_pattern(data["D"])
    Oy = get_Oy(data_ptrn)

    PPP_vals = np.empty((nsim, 2))
    for m_ind in tqdm(range(nsim)):
        m = skip_step * m_ind
        # compute Dy
        pi = get_probs(data, ps, m)
        Ey = get_Ey(data_ptrn, pi, data["N"])
        Dy = get_Dy(Oy, Ey, data_ptrn)

        # compute Dy
        ppdata = bernoulli.rvs(get_probs(data, ps, m))
        ppddata_ptrn = to_str_pattern(ppdata)

        Oystr = get_Oy(ppddata_ptrn)
        Eystr = get_Ey(ppddata_ptrn, pi, data["N"])
        Dystr = get_Dy(Oystr, Eystr, ppddata_ptrn)

        PPP_vals[m_ind, 0] = sum(Dy.values())
        PPP_vals[m_ind, 1] = sum(Dystr.values())

    return PPP_vals


def compute_brier_individual(probs_dict, obs):
    p_obs = probs_dict[obs]
    probs = np.array(list(probs_dict.values()))
    probs_sum = np.sum(probs ** 2) - (p_obs ** 2)
    score = ((1.0 - p_obs) ** 2) + probs_sum
    return score


def compute_log_score(probs_dict, data):
    n = data.shape[0]
    score = 0.0
    for i in range(n):
        score_individual = -np.log(probs_dict[data[i]])
        score = score + score_individual
    return score

def get_g2_score(data_ptrn, post_y, N):
    Oy = get_Oy(data_ptrn)
    Ey = get_Ey(data_ptrn, expit(post_y), N)
    Dy = get_Dy(Oy, Ey, data_ptrn)
    return sum(Dy.values())


def get_logscore1(data_ptrn, post_y):
    E_prob = get_response_probs(data_ptrn, expit(post_y))
    return compute_log_score(E_prob, data_ptrn)


def get_logscore2(data_ptrn, post_y):
    distinct_patterns = np.unique(data_ptrn)
    E_prob = get_response_probs(data_ptrn, expit(post_y))
    Oy = get_Oy(data_ptrn)
    scores = 0.0
    for ptrn in distinct_patterns:
        lgscr = Oy[ptrn] * np.log(E_prob[ptrn])
        scores = scores - lgscr
    return scores


def get_brier_score(data_ptrn, post_y):
    E_prob = get_response_probs(data_ptrn, expit(post_y))
    n = data_ptrn.shape[0]
    score = 0.0
    for i in range(n):
        score_i = compute_brier_individual(E_prob, data_ptrn[i])
        score += score_i
    return score


def adjust_beta_sign(ps):
    num_samples = ps["alpha"].shape[0]
    ps["beta_rot"] = ps["beta"].copy()
    # if 'Phi_cov' in ps.keys():
    #     ps['Phi_cov_rot'] = ps['Phi_cov'].copy()
    for i in range(num_samples):
        sign1 = np.sign(ps["beta"][i, 0, 0])
        sign2 = np.sign(ps["beta"][i, 3, 1])
        ps["beta_rot"][i, :3, 0] = ps["beta"][i, :3, 0] * sign1
        ps["beta_rot"][i, 3:, 1] = ps["beta"][i, 3:, 1] * sign2

        # if 'Phi_cov' in ps.keys():
        #     ps['Phi_cov_rot'][i,0,1] = sign1 * sign2 * ps['Phi_cov'][i,0,1]
        #     ps['Phi_cov_rot'][i,1,0] = ps['Phi_cov'][i,0,1]

    ps["beta"] = ps["beta_rot"].copy()
    # if 'Phi_cov' in ps.keys():
    #     ps['Phi_cov'] = ps['Phi_cov_rot'].copy()
    return ps


def get_method1(ps, dim_K, nsim):
    m_alpha = ps["alpha"].mean(axis=0)
    if "Marg_cov" in ps.keys():
        m_Marg_cov = ps["Marg_cov"].mean(axis=0)
        post_y = multivariate_normal.rvs(mean=m_alpha, cov=m_Marg_cov, size=nsim)
    else:
        m_beta = ps["beta"].mean(axis=0)
        if "Phi_cov" in ps.keys():
            m_Phi_cov = ps["Phi_cov"].mean(axis=0)
        else:
            m_Phi_cov = np.eye(dim_K)
        zz_from_prior = multivariate_normal.rvs(
            mean=np.zeros(dim_K), cov=m_Phi_cov, size=nsim
        )
        post_y = m_alpha + zz_from_prior @ m_beta.T
    return post_y


def get_method2(ps, dim_J, dim_K, nsim, skip_step):
    post_y = np.empty((nsim, dim_J))
    for m_ind in tqdm(range(nsim)):
        m = skip_step * m_ind
        m_alpha = ps["alpha"][m]
        if "Marg_cov" in ps.keys():
            m_Marg_cov = ps["Marg_cov"][m]
            post_y_sample = multivariate_normal.rvs(
                mean=m_alpha, cov=m_Marg_cov, size=1
            )
        else:
            m_beta = ps["beta"][m]
            if "Phi_cov" in ps.keys():
                m_Phi_cov = ps["Phi_cov"][m]
            else:
                m_Phi_cov = np.eye(dim_K)
            zz_from_prior = multivariate_normal.rvs(
                mean=np.zeros(dim_K), cov=m_Phi_cov, size=1
            )
            post_y_sample = m_alpha + zz_from_prior @ m_beta.T
        post_y[m_ind] = post_y_sample
    return post_y


def get_scores(ps, data, nsim, method_num=2):

    mcmc_length = ps["alpha"].shape[0]
    dim_J = ps['alpha'].shape[1]
    dim_K = ps["beta"].shape[-1]
    if nsim>mcmc_length:
        print('nsim > posterior sample size')
        print('Using nsim = %d'%mcmc_length)
        nsim = mcmc_length
    skip_step = int(mcmc_length/nsim)

    data_ptrn = to_str_pattern(data["test"]["DD"])

    if method_num == 1:
        # fix at posterior mean
        ps = adjust_beta_sign(ps)
        post_y = get_method1(ps, dim_K, nsim)
    elif method_num == 2:
        # fix use whole distribution
        post_y = get_method2(ps, dim_J, dim_K, nsim, skip_step)
    else:
        print("method_num not found")
    set_trace()
    scores = dict()
    g2_score = get_g2_score(data_ptrn, post_y, data["test"]["N"])
    lgscr1 = get_logscore1(data_ptrn, post_y)
    lgscr2 = get_logscore2(data_ptrn, post_y)
    diff = g2_score - lgscr1
    brier_score = get_brier_score(data_ptrn, post_y)
    print("G2 = %.2f" % g2_score)
    print("logscore = %.2f" % lgscr1)
    print("logscore2 = %.2f" % lgscr2)
    print("G2-logscore = %.2f" % diff)
    print("brier = %.2f" % brier_score)
    print("\n\n")

    scores["g2"] = g2_score
    scores["logscore"] = lgscr1
    scores["brier"] = brier_score
    return scores
