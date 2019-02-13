from scipy.stats import multivariate_normal, invwishart, invgamma, norm
from numpy.linalg import inv
from scipy.special import expit
import numpy as np
import sys
from time import sleep
from scipy.special import logsumexp

def prior_w(k, j, mu0=None, Sigma0=None, size=1, random_seed = None):
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

    ww = norm.rvs(size=k * j * size ).reshape((size, j,k,))
    ww[0,1] = 0.

    return ww


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


# def construct_Sigma(sigma):
#
#
#     Sigma = np.empty_like(R)
#
#     if sigma.ndim == 1:
#         return np.diag(sigma[0])
#
#     for i in range(Sigma.shape[0]):
#         D = np.diag(sigma[i])
#         Sigma[i] = D @ R[i] @ D
#
#     return Sigma


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

#
# def log_weight(data, params, nsim_z):
#     """
#     Returns the logweight of one particle.
#     data is supposed to include a single point data['y]
#     """
#     Kc = data['Kc']
#     Kb = data['Kb']
#
#
#     # generate $\{z_i\}_{i=1}^M \sim p(z| y^{(c)}, \theta)$
#     mu_bar, Sigma_bar = norm_cond_1(data['y'][:Kc],
#             params['mu'], params['Sigma'], Kc, Kb)
#
#     z = multivariate_normal.rvs(mean=mu_bar,
#                     cov = Sigma_bar, size=nsim_z).reshape(nsim_z, Kb)
#
#     # Compute $p(y^{(b)} | z_i)$
#     logv = loglklhd_z(data['y'][Kc:], z)
#     # Efficient alternative to logp1 = np.log(np.average(np.exp(logv)))
#     logp1 = logsumexp(logv) - np.log(logv.shape[0])
#
#
#     # Compute  p(y^{(c)}| \theta)
#     logp2 = multivariate_normal.logpdf(data['y'][:Kc], mean = params['mu'][:Kc],
#                                   cov = params['Sigma'][:Kc, :Kc])
#
#     logweight = logp1 + logp2
#
#     return logweight
#
#
# def get_weights(data, particles, nsim_z=1000):
#     logweights = np.empty(particles['N'])
#
#     for m in range(particles['N']):
#         # for each particle
#         logweights[m]= log_weight(data,
#             {'mu':particles['mu'][m], 'Sigma':particles['Sigma'][m]},
#             nsim_z=nsim_z)
#
#     return logweights


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
