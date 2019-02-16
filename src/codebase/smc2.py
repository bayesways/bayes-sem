from scipy.stats import multivariate_normal, invwishart, invgamma, norm
from numpy.linalg import inv
from scipy.special import expit
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


def loglklhd_z(y, z):
    """
    dim(y) = Kb
    dim(z) = Nb, Kb

    output p1 needs to be dim(p1) = Nb
    """
    try:
        a = np.log(expit(z)) * y + np.log(1 - expit(z)) * (1-y)
        lglkhd = np.sum(a, axis=1)
    except:
        lglkhd = 0
    return lglkhd


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


def jitter(data, particles, nsim_z=1000):
    size = particles['N']
    for j in range(size):
        jittered_sample = mcmc(data,
            params = {
            'mu':particles['mu'][j],
            'sigma': particles['sigma'][j],
            'R' : particles['R'][j],
            'Sigma' : particles['Sigma'][j]
            },
          nsim=20, nsim_z=nsim_z)

        particles['mu'][j] = jittered_sample['mu']
        particles['sigma'][j] = jittered_sample['sigma']
        particles['R'][j] = jittered_sample['R']
        particles['Sigma'][j] = jittered_sample['Sigma']

    return particles


def log_weight(data, params):
    """
    Returns the logweight of one particle.
    data is supposed to include a single point data['y']
    """

    Sigma_temp = np.diag(params['sigma'])
    Omega = params['beta'] @ params['beta'].T + Sigma_temp
    logweight = multivariate_normal.logpdf(data['y'],
        mean=np.zeros(data['J']), cov= Omega )

    return logweight


def sample_beta(data, Sigma, nsim_z):
    """
    Version 1
    Sample z, one row at a time
    """

    output_beta = np.empty((data['N'], data['K']))

    weights = np.empty(nsim_z)

    beta_temp = norm.rvs(size=(nsim_z *data['K'] * data['J'])).reshape(nsim_z,
        data['J'], data['K'])

    for j in range(nsim_z):
        beta_temp[j,0,:] = trunc_normal(0, np.zeros(2), np.eye(2))
        beta_temp[j,0,1] = 0.
        beta_temp[j,1,:] = trunc_normal(1, np.zeros(2), np.eye(2))

    for j in range(nsim_z):
        Omega = beta_temp[j] @ beta_temp[j].T + Sigma
        weights[j] = np.sum(multivariate_normal.logpdf(data['y'],
            mean=np.zeros(data['J']), cov= Omega ))

        # equivalent to above
        # weights[j] = matrix_normal.logpdf(data['y'],
        #             mean = np.tile(np.zeros(data['J']), data['N']).\
        #                 reshape((data['N'],data['J'])),
        #             rowcov=np.eye(data['N']),
        #             colcov=Omega)

    return sample_from_weighted_array(beta_temp, weights)



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
