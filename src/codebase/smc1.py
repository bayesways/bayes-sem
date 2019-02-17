from scipy.stats import multivariate_normal, invwishart, invgamma, norm
from numpy.linalg import inv
from scipy.special import expit
from .pseudogibbs2 import create_w_columns,\
    sample_beta, sample_from_weighted_array
from .pseudogibbs import trunc_normal
import numpy as np
import sys
from time import sleep
from scipy.special import logsumexp


def trunc_normal_zero(i, mean, cov, max_counter=10):
    """
    Return a vector x ~ N(m, C) where x[i] >0 and x[j]=0 for j>i
    """
    if mean.shape[0] == 1:
        a = 0.
        counter = 0
        while a<=0.:
            if counter >= max_counter:
                print("No sample")
                return np.array(0.)
            else:
                a = multivariate_normal.rvs(mean, cov)
                counter+=1
        return np.array(a)

    else:
        a = np.zeros(mean.shape[0])
        counter = 0
        while a[i]<=0.:
            if counter >= max_counter:
                # print("No sample")
                return np.zeros(mean.shape[0])
            else:
                a = multivariate_normal.rvs(mean, cov)
                counter += 1
        a[(i+1):] = 0.
        return np.array(a)


def prior_beta(k, j, mu0=None, Sigma0=None, size=1, random_seed = None):
    """
    returns mean vectors (size x dim)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # if mu0 is None:
    #     mu0 = np.zeros(dim)
    #
    # if Sigma0 is None:
    #     Sigma0 = np.eye(dim)*1e2

    beta = norm.rvs(size=k * j * size ).reshape((size, j,k,))
    for j in range(size):
        beta[j,0,:] = trunc_normal_zero(0, np.zeros(2), np.eye(2))
        # beta[j,0,1] = 0.
        beta[j,1,:] = trunc_normal_zero(1, np.zeros(2), np.eye(2))

    return beta


def prior_sigma(dim, a0 = None, b0 = None, size=1, random_seed = None):
    """
    returns variance vector (size x dim)
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    if a0 is None:
        a0 = 1e-0

    if b0 is None:
        b0 = 1e-0

    s = np.empty((size, dim))
    for i in range(dim):
        s[:,i] = invgamma.rvs(a0, scale= b0, size=size)

    if size ==1:
        s = s[0]
    return s


def construct_Sigma(sigma):
    if sigma.ndim == 1:
        return np.diag(sigma[0])

    else:
        n = sigma.shape[0]
        k = sigma.shape[1]
        Sigma = np.empty((n, k, k ))
        for i in range(n):
            Sigma[i] = np.diag(sigma[i])

    return Sigma


def multinomial_sample_particles(particles, probs = None):
    size = particles['N']

    # if no weights assign uniform weights
    if probs is None:
        probs = np.ones(size)

    assert size == len(probs)

    # normalize weights if necessary
    if np.sum(probs) != 1:
        normalized_probs = probs / np.sum(probs)
    sampled_index = np.random.choice(np.arange(size),
                                     p=normalized_probs,
                                     size=size)

    for key in particles['parameter_names']:
            particles[key] = particles[key][sampled_index]

    return particles


def mcmc(data, params, nsim=100, nsim_b = 100):

    beta_s = np.empty((nsim, data['J'], data['K']))
    sigma_s = np.empty((nsim, data['J']))
    z_s = np.empty((nsim, data['N'], data['K']))


    zz_temp = data['z'].copy()

    sigma_temp = params['sigma'].copy()
    # sigma_temp = data['sigma'].copy()
    Sigma_temp = np.diag(sigma_temp)
    beta_temp = params['beta'].copy()

    C0 = 1e2
    mu0 = 0
    prior_a = 1e-1
    prior_b = 1e-1


    for j in range(nsim):
        # sample z
        zz_temp = np.empty((data['N'], data['K']))
        inv_Sigma = inv(Sigma_temp)
        cov1 = inv(np.eye(data['K']) +  beta_temp.T @ inv_Sigma @ beta_temp)
        for t in range(data['N']):
            mean = cov1 @ beta_temp.T @ inv_Sigma @ data['y'][t]
            zz_temp[t] = multivariate_normal.rvs(mean, cov1)
        z_s[j] = zz_temp

        # sample sigma
        for i in range(data['J']):
            sigma_a = (data['N'] + prior_a )*.5
            aux = data['y'][:,i] - zz_temp @ create_w_columns(i,k=2, ww=beta_temp).T
            d = aux.T @ aux
            sigma_b = (prior_a*(prior_b**2)+d)* 0.5
            sigma_temp[i] = invgamma.rvs(sigma_a, scale = sigma_b)
        sigma_s[j] = sigma_temp
        Sigma_temp = np.diag(sigma_temp)


        # sample w
        for i in range(data['J']):
            if (i+1)<=data['K']:
                aux1= zz_temp[:,:(i+1)].T @ zz_temp[:,:(i+1)]
                inv_C = C0**(-1)*np.eye(i+1) + sigma_temp[i]**(-2)*aux1
                C = inv(inv_C)
                aux2= zz_temp[:,:(i+1)].T @ data['y'][:,i]
                mean = C @ (C0**(-1)*mu0*np.ones(i+1) + sigma_temp[i]**(-2)*aux2 )
                if mean[i] > 0:
                    beta_temp[i,:(i+1)] = trunc_normal(i, mean, C)

            else:
                aux1= zz_temp.T @ zz_temp
                inv_C = C0**(-1)*np.eye(data['K']) + sigma_temp[i]**(-2)*aux1
                C = inv(inv_C)
                aux2= zz_temp.T @ data['y'][:,i]
                mean = C @ (C0**(-1)*mu0*np.ones(data['K'])+sigma_temp[i]**(-2)*aux2 )
                beta_temp[i,:] = multivariate_normal.rvs(mean, cov=C)
        beta_s[j] = beta_temp

    output = dict()
    output['beta'] = beta_s[-1]
    output['sigma'] = sigma_s[-1]
    # output['Sigma'] = Sigma_s[-1]
    return output


#
# def mcmc(data, params, nsim=100, nsim_b = 100):
#
#     beta_s = np.empty((nsim, data['J'], data['K']))
#     sigma_s = np.empty((nsim, data['J']))
#     Sigma_s = np.empty((nsim, data['K'], data['K']))
#     z_s = np.empty((nsim, data['N'], data['K']))
#
#
#     zz_temp = data['z'].copy()
#
#     sigma_temp = params['sigma'].copy()
#     # sigma_temp = data['sigma'].copy()
#     Sigma_temp = np.diag(sigma_temp)
#     beta_temp = params['beta'].copy()
#
#     C0 = 1e2
#     mu0 = 0
#     prior_a = 1e-1
#     prior_b = 1e-1
#
#     for j in range(nsim):
#
#         # sample sigma
#         for i in range(data['J']):
#             sigma_a = (data['N'] + prior_a )*.5
#             aux = data['y'][:,i] - zz_temp @ create_w_columns(i,k=2,\
#                 ww=beta_temp).T
#             d = aux.T @ aux
#             sigma_b = (prior_a*(prior_b**2)+d)/2.
#             sigma_temp[i] = invgamma.rvs(sigma_a, scale = sigma_b)
#         sigma_s[j] = sigma_temp
#         Sigma_temp = np.diag(sigma_temp)
#
#
#         # sample w
#         beta_temp = sample_beta(data, Sigma_temp, nsim_b)
#         beta_s[j] = beta_temp
#
#     output = dict()
#     output['beta'] = beta_s[-1]
#     output['sigma'] = sigma_s[-1]
#     # output['Sigma'] = Sigma_s[-1]
#     return output



def jitter(data, particles, nsim_b=1000):
    size = particles['N']
    for j in range(size):
        jittered_sample = mcmc(data,
            params = {
            'beta':particles['beta'][j],
            'sigma': particles['sigma'][j]
            },
          nsim=20, nsim_b=nsim_b)

        particles['beta'][j] = jittered_sample['beta']
        particles['sigma'][j] = jittered_sample['sigma']
        # particles['Sigma'][j] = jittered_sample['Sigma']

    return particles


def log_weight(data, params):
    """
    Returns the logweight of one particle.
    data is supposed to include a single point data['y']
    """

    zz_temp = norm.rvs(size=(nsim_z *data['K'])).reshape(nsim_z,
        data['K'])

    mean = zz_temp @ params['beta'].T
    Sigma_temp = np.diag(params['sigma'])
    logweight = multivariate_normal.logpdf(data['y'], mean=mean,
            cov=Sigma_temp)

    return logweight


def get_weights(data, particles, nsim_z=1000):
    logweights = np.empty(particles['N'])

    for m in range(particles['N']):
        # for each particle
        logweights[m]= log_weight(data,
            {'beta':particles['beta'][m], 'sigma':particles['sigma'][m]})

    return logweights


def ESS(w):
    a = logsumexp(w[~np.isnan(w)])*2
    b = logsumexp(2*w[~np.isnan(w)])
    return  np.exp(a-b)

#
# def multinomial_sample_z(particles, probs = None):
#     size = particles.shape[0]
#
#     # if no weights assign uniform weights
#     if probs is None:
#         probs = np.ones(size)
#
#     assert size == len(probs)
#
#     # normalize weights if necessary
#     if np.sum(probs) != 1:
#         normalized_probs = probs / np.sum(probs)
#     sampled_index = np.random.choice(np.arange(size),
#                                      p=normalized_probs)
#
#     return particles[sampled_index]
