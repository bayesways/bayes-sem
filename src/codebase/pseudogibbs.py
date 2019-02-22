import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal, invwishart, invgamma
from numpy.linalg import inv
from pdb import set_trace
from time import sleep
import sys

def trunc_normal(i, mean, cov, max_counter=10):
    if mean.shape[0] == 1:
        a = 0.
        counter = 0
        while a<=0.:
            if counter >= max_counter:
                # print("No sample")
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
        return np.array(a)


def create_w_columns(i,k,ww):
    if i+1<=k:
        a = np.zeros(k)
        a[:(i+1)] = ww[i,:(i+1)]
        return a.T
    else:
        return ww[i,:]


def mcmc(data, nsim=100, nsim_z = 10, visual_bar=True):

    w_s = np.empty((nsim, data['J'], data['K']))
    sigma_s = np.empty((nsim, data['J']))
    Sigma_s = np.empty((nsim, data['K'], data['K']))
    z_s = np.empty((nsim, data['N'], data['K']))

    # sigma_temp = np.ones(data['J'])
    sigma_temp = data['sigma'].copy()
    Sigma_temp = np.diag(sigma_temp)

    # zz_temp = data['z'].copy()
    ww_temp = data['beta'].copy()
    # ww_temp = norm.rvs(size=data['J']*data['K']).reshape((data['J'],data['K']))
    # ww_temp[0,1] = 0.

    C0 = 1e2
    mu0 = 0
    prior_a = 1e-1
    prior_b = 1e-1



    # Auxialiry variables
    if visual_bar:
        bars = 0
        bar_num = int(nsim/10)
        if bar_num < 1:
            bar_num = 1


    for j in range(nsim):

        # sample z
        zz_temp = sample_z(data, ww_temp, Sigma_temp, nsim_z)
        z_s[j] = zz_temp


        # sample sigma
        for i in range(data['J']):
            sigma_a = (data['N'] + prior_a )*.5
            aux = data['y'][:,i] - zz_temp @ create_w_columns(i,k=2, ww=ww_temp).T
            d = aux.T @ aux
            sigma_b = (prior_a*(prior_b**2)+d)/2.
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
                    ww_temp[i,:(i+1)] = trunc_normal(i, mean, C)

            else:
                aux1= zz_temp.T @ zz_temp
                inv_C = C0**(-1)*np.eye(data['K']) + sigma_temp[i]**(-2)*aux1
                C = inv(inv_C)
                aux2= zz_temp.T @ data['y'][:,i]
                mean = C @ (C0**(-1)*mu0*np.ones(data['K'])+sigma_temp[i]**(-2)*aux2 )
                ww_temp[i,:] = multivariate_normal.rvs(mean, cov=C)
        w_s[j] = ww_temp


        if visual_bar:
            if j%bar_num == 0:
                bars +=1
                sys.stdout.write('\r')
                sys.stdout.write("[%-10s]" % ('='*bars))
                sys.stdout.write(" "+str(j)+" steps")
                sys.stdout.flush()
                sleep(0.25)

    output = dict()
    output['beta'] = w_s
    output['z'] = z_s
    output['sigma'] = sigma_s
    output['Sigma'] = Sigma_s


    return output


def sample_z(data, ww, Sigma, nsim_z):
    """
    Version 1
    Sample z, one row at a time
    """

    output_z = np.empty((data['N'], data['K']))

    for n in range(data['N']):
        weights = np.empty(nsim_z)

        zz_temp = norm.rvs(size=(nsim_z *data['K'])).reshape(nsim_z,
            data['K'])

        mean = zz_temp @ ww.T

        for j in range(nsim_z):
            weights[j] = multivariate_normal.pdf(data['y'][n], mean=mean[j],
                cov=Sigma)

        output_z[n] = sample_from_weighted_array(zz_temp, weights)
    return output_z

#
# def sample_z2(data, ww, Sigma, nsim_z):
#     """
#     Version 2
#     Sample z all rows at once
#     """
#     z = norm.rvs(size=(nsim_z * data['N']*data['K'])).reshape(nsim_z,
#         data['N'], data['K'])
#
#     weights = np.empty(nsim_z)
#     for i in range(nsim_z):
#         mean = z[i]@ww.T
#         # weights[i] = np.sum(multivariate_normal.logpdf(data['y'],
#         #     mean=mean, cov=Sigma ))
#
#         # equivalent to above
#         weights[j] = matrix_normal.logpdf(data['y'],
#                     mean = mean,
#                     rowcov=np.eye(data['N']),
#                     colcov=Sigma)
#
#     return sample_from_weighted_array(z, weights)



def sample_index(probs, size=1):
    """
    Sample i from [1,...,N] where P(i) = probs[i]
    """
    # normalize weights if necessary
    if np.sum(probs) != 1:
        normalized_probs = probs / np.sum(probs)
    a = np.random.choice(len(probs), p=normalized_probs)

    return a


def sample_from_weighted_array(v, probs, size=1):
    """
    Sample v[i] from v where P(v[i]) = probs[i]
    """
    ind = sample_index(probs, size=size)
    return v[ind]
