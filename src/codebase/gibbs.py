import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal, invwishart, invgamma
from numpy.linalg import inv
from pdb import set_trace

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


def mcmc(data, nsim=100):

    w_s = np.empty((nsim, data['J'], data['K']))
    sigma_s = np.empty((nsim, data['J']))
    z_s = np.empty((nsim, data['N'], data['K']))

    sigma_temp = np.ones(data['J'])
    # sigma_temp = data['sigma'].copy()
    Sigma_temp = np.diag(sigma_temp)

    zz_temp = data['z'].copy()
    # ww_temp = data['w'].copy()
    ww_temp = norm.rvs(size=data['J']*data['K']).reshape((data['J'],data['K']))
    ww_temp[0,1] = 0.

    C0 = 1e2
    mu0 = 1
    prior_a = 1e-1
    prior_b = 1e-1

    for j in range(nsim):
        # sample z
        zz_temp = np.empty((data['N'], data['K']))
        inv_Sigma = inv(Sigma_temp)
        cov1 = inv(np.eye(data['K']) +  ww_temp.T @ inv_Sigma @ ww_temp)
        for t in range(data['N']):
            mean = cov1 @ ww_temp.T @ inv_Sigma @ data['y'][t]
            zz_temp[t] = multivariate_normal.rvs(mean, cov1)
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

    output = dict()
    output['w'] = w_s
    output['z'] = z_s
    output['sigma'] = sigma_s

    return output
