from numpy.linalg import inv,cholesky
import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, bernoulli
from scipy.special import expit, logit


def check_posdef(R):
    """
    Check that matrix is positive definite
    Inputs
    ============
    - matrix
    """
    try:
        cholesky(R)
    except NameError  :
        print("\n Error: could not compute cholesky factor of R")
        raise
        sys.exit(1)
    return 0


def extract_values_above_diagonal(M, offset = 1):
    """
    extract the matrix entries that sit above the
    diagonal. To include the diagonal set offset=0,
    to move further up use offest>1.
    Inputs
    ============
    - M
    - offset
    Output
    ============
    - flatten np.array of values
    """
    return M[np.triu_indices( n = M.shape[0], k =offset)]


def flatten_df(df0, val_name, var_name = 'K'):
    df = pd.DataFrame(df0)
    df.columns  = df.columns + 1
    df['J'] = np.arange(len(df))+1
    return df.melt(id_vars=['J'], var_name=var_name, value_name = val_name)


def post_summary(samples):
    mean =  pd.DataFrame(np.mean(samples, axis=0))
    ps_df = flatten_df(mean, 'mean')
    median = pd.DataFrame(np.median(samples, axis=0))
    ps_df['median'] = flatten_df(median, 'median')['median']
    per1 = pd.DataFrame(np.percentile(samples, 2.5,axis=0))
    ps_df['q2.5'] = flatten_df(per1, 'q2.5')['q2.5']
    per2 = pd.DataFrame(np.percentile(samples, 97.5,axis=0))
    ps_df['q97.5'] = flatten_df(per2, 'q97.5')['q97.5']
    return ps_df


def C_to_R(M):
    """
    Send a covariance matrix M to the corresponding
    correlation matrix R
    Inputs
    ============
    - M : covariance matrix
    Output
    ============
    - correlation matrix
    """
    d = np.asarray(M.diagonal())
    d2 = np.diag(d**(-.5))
    R = d2 @ M @ d2
    return R


def gen_data0(nsim_data, J=6, rho = 0.3, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.array([1,.3,.5, -1.2, -.5, 1])
    Theta_corr = np.eye(J)
    for i in [1,2,5]:
        for j in [3,4]:
            Theta_corr[i,j] = rho
            Theta_corr[j,i] = rho
    Theta = Theta_corr

    assert check_posdef(Theta)==0
    yy = multivariate_normal.rvs(mean = alpha, cov=Theta, size=nsim_data)

    DD = bernoulli.rvs(p=expit(yy))

    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['J'] = J
    data['alpha'] = alpha
    # data['beta'] = beta
    data['Theta'] = Theta
    data['rho'] = rho
    data['y'] = yy
    data['D'] = DD

    return(data)


def gen_data(nsim_data, J=6, c=1,
             off_diag_residual = False, off_diag_corr = 0.2,
             random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    # beta = np.array([[1,0],
    #                  [b, 0],
    #                  [b,0],
    #                  [0,1],
    #                  [0,b],
    #                  [0,b]], dtype=float)
    #
    # sigma_z = np.repeat(np.sqrt(c), K)
    # Phi_corr = np.eye(K)
    # Phi_corr[0,1] = rho
    # Phi_corr[1,0] = rho
    # Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)
    #
    # sigma_sq = 1 - np.diag(beta @ Phi_cov @ beta.T)
    # sigma = np.sqrt(sigma_sq)

    sigma_sq = np.repeat(c,J)
    sigma = np.sqrt(sigma_sq)

    if off_diag_residual:
        Theta_corr = np.eye(J)
#         Theta = np.diag(sigma_sq)
        for i in [1,2,5]:
            for j in [3,4]:
                Theta_corr[i,j] = off_diag_corr
                Theta_corr[j,i] = off_diag_corr
        Theta = np.diag(sigma) @ Theta_corr @  np.diag(sigma)
    else:
        Theta = np.diag(sigma_sq)

    assert check_posdef(Theta)==0
    yy = multivariate_normal.rvs(mean = alpha, cov=Theta, size=nsim_data)

    DD = bernoulli.rvs(p=expit(yy))

    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['J'] = J
    data['alpha'] = alpha
    # data['beta'] = beta
    data['Theta'] = Theta
    data['Theta_corr']  = Theta_corr
    data['sigma'] = sigma
    data['off_diag_residual'] = off_diag_residual
    data['off_diag_corr'] = off_diag_corr
    data['y'] = yy
    data['D'] = DD

    return(data)


def flatten_corr_matrix_samples(Rs, colnames=None):
    """
    Flatten a [N, K, K ] array of correlation
    matrix samples to a [N,M] array where
    M is the number of of elements below the
    diagonal for a K by K matrix.
    For each sample correlation matrix we care only
    for these M parameters
    Inputs
    ============
    - Rs : samples to flattent out, should be
        of dimension [N,K,K]
    - N : number of samples
    Output
    ============
    -  a dataframe of size [N,M]
    """

    N,K = Rs.shape[0], Rs.shape[1]
    if colnames is None:
        colnames = [str(x) for x in range(K)]

    assert len(colnames) == K, 'colnames should as long as the columns of R'
    cnames = corr_headers(colnames)

    M = len(cnames)
    k=0
    fRs = np.empty((N,M))
    for i in range (N):
        fRs[i,:] = flatten_corr(Rs[i,:,:])

    fRs = pd.DataFrame(fRs)
    fRs.columns=cnames


    return fRs

def corr_headers(colnames):
    """
    Return the headers for the DataFrame
    of the correlation coefficients.
    If the original correlation matrix is
    of size [K, K ], then corr_headers
    takes in K headers and returns an
    array of size [M,], where
    M is the number of of elements above the
    diagonal for a K by K matrix.
    To be used together with `flatten_corr`.
    Start with a correlation matrix
    R, get the correlation coefficients with
    flatten_corr(R) and turn it into a
    dataframe with columns headers
    corr_headers(colnames) where
    colnames are the headers of the data.
    Inputs
    ============
    - R : headers of the data.
    Output
    ============
    -  an array of size [M,]
    """
    colnames = np.array(colnames)
    dfcols = list(zip(colnames[np.triu_indices(colnames.shape[0], k=1)[0]],\
                 colnames[np.triu_indices(colnames.shape[0], k=1)[1]]))
    return dfcols


def flatten_corr(a):
    """
    Flatten a [K, K ] correlation
    matrix to [M,] array where
    M is the number of of elements above the
    diagonal for a K by K matrix.
    Inputs
    ============
    - R : matrix to flattent out, should be
        of dimension [K,K]
    Output
    ============
    -  an array of size [M,]
    """
    return a[np.triu_indices(a.shape[0], k=1)]
