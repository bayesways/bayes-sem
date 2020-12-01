import numpy as np
import pandas as pd
import pystan
from scipy.stats import norm, multivariate_normal, bernoulli
from scipy.special import expit, logit
from numpy.linalg import inv, cholesky


def flatten_df(df0, val_name, var_name='K'):
    df = pd.DataFrame(df0)
    df.columns = df.columns + 1
    df['J'] = np.arange(len(df))+1
    return df.melt(id_vars=['J'], var_name=var_name, value_name=val_name)


def post_summary(samples):
    mean = pd.DataFrame(np.mean(samples, axis=0))
    ps_df = flatten_df(mean, 'mean')
    median = pd.DataFrame(np.median(samples, axis=0))
    ps_df['median'] = flatten_df(median, 'median')['median']
    per1 = pd.DataFrame(np.percentile(samples, 2.5, axis=0))
    ps_df['q2.5'] = flatten_df(per1, 'q2.5')['q2.5']
    per2 = pd.DataFrame(np.percentile(samples, 97.5, axis=0))
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


def gen_data(nsim_data, J=6, K=2, rho=0.2, c=0.65, b=0.8,
             off_diag_residual=False, off_diag_corr=0.2,
             cross_loadings=False,
             random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    if cross_loadings:
        beta = np.array(
            [[1, 0],
            [b, 0],
            [b, .6],
            [.6, 1],
            [0, b],
            [0, b]],
            dtype=float
            )
    else:
        beta = np.array([[1, 0],
                         [b, 0],
                         [b, 0],
                         [0, 1],
                         [0, b],
                         [0, b]], dtype=float)

    sigma_z = np.repeat(np.sqrt(c), K)
    Phi_corr = np.eye(K)
    Phi_corr[0, 1] = rho
    Phi_corr[1, 0] = rho
    Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)

    sigma = np.ones(J)

    if off_diag_residual:
        Theta_corr = np.eye(J)
        for i in [1, 2, 5]:
            for j in [3, 4]:
                Theta_corr[i, j] = off_diag_corr
                Theta_corr[j, i] = off_diag_corr
        Theta = np.diag(sigma) @ Theta_corr @  np.diag(sigma)
    else:
        Theta = np.diag(sigma**2)

    Marg_cov = beta @ Phi_cov @ beta.T + Theta
    yy = multivariate_normal.rvs(mean=alpha, cov=Marg_cov, size=nsim_data)

    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['sigma_z'] = sigma_z
    data['Phi_corr'] = Phi_corr
    data['Phi_cov'] = Phi_cov
    data['Marg_cov'] = Marg_cov
    data['Theta'] = Theta
    data['sigma'] = sigma
    data['y'] = yy
    data['off_diag_residual'] = off_diag_residual
    data['cross_loadings'] = cross_loadings

    return(data)


def gen_data_binary(nsim_data, J=6, K=2, rho=0.2, b=0.8,
                    off_diag_residual=False, rho2=0.1, c=1,
                    cross_loadings=False, cross_loadings_level=3,
                    method=3, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    if cross_loadings:
        if cross_loadings_level == 0:
            beta = np.array([[1, 0.2],
                             [b, -0.3],
                             [b, -.05],
                             [-0.2, 1],
                             [-.08, b],
                             [0.15, b]], dtype=float)
        elif cross_loadings_level == 1:
            beta = np.array([[1, 0],
                             [b, 0],
                             [b, .4],
                             [0, 1],
                             [.4, b],
                             [0, b]], dtype=float)
        elif cross_loadings_level == 2:
            beta = np.array([[1, 0],
                             [b, .4],
                             [b, .4],
                             [0, 1],
                             [.4, b],
                             [.4, b]], dtype=float)
        elif cross_loadings_level == 3:
            beta = np.array([[1, 0],
                             [b, 0],
                             [b, .6],
                             [.6, 1],
                             [0, b],
                             [0, b]], dtype=float)
        else:
            print('Noisy Level should be in [0,1,2]')
    else:
        beta = np.array([[1, 0],
                         [b, 0],
                         [b, 0],
                         [0, 1],
                         [0, b],
                         [0, b]], dtype=float)

    alpha = np.zeros(J)

    sigma_z = np.ones(K)
    Phi_corr = np.eye(K)
    Phi_corr[0, 1] = rho
    Phi_corr[1, 0] = rho
    # Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)
    Phi_cov = Phi_corr

    assert check_posdef(Phi_cov) == 0
    zz = multivariate_normal.rvs(mean=np.zeros(K), cov=Phi_cov,
                                 size=nsim_data)

    sigma_theta = np.ones(J)*c
    Theta_corr = np.eye(J)
    if off_diag_residual:
        for i in [1, 2, 5]:
            for j in [3, 4]:
                Theta_corr[i, j] = rho2
                Theta_corr[j, i] = rho2
        Theta = np.diag(sigma_theta) @ Theta_corr @  np.diag(sigma_theta)
    else:
        Theta = np.diag(sigma_theta**2)
    if method == 1:  # logit method
        yy = alpha + zz @ beta.T
        DD = bernoulli.rvs(p=expit(yy))
    elif method == 2:  # probit method
        yy = alpha + zz @ beta.T
        DD = bernoulli.rvs(p=norm.cdf(yy))
    elif method == 3:  # logit2 method
        assert check_posdef(Theta) == 0
        ee_seed = multivariate_normal.rvs(mean=np.zeros(J), cov=Theta,
                                          size=nsim_data)
        ee = logit(norm.cdf(ee_seed))
        yy = alpha + zz @ beta.T + ee
        DD = (yy > 0).astype(int)
    elif method == 4:  # probit2 method
        assert check_posdef(Theta) == 0
        ee = multivariate_normal.rvs(mean=np.zeros(J), cov=Theta,
                                     size=nsim_data)

        yy = alpha + zz @ beta.T + ee
        DD = (yy > 0).astype(int)
    else:
        print("method must be in [1:4]")

    if method in [1, 3]:
        model_type = 'logit'
    elif method in [2, 4]:
        model_type = 'probit'
    else:
        print("data method must be in [1:4]")

    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['sigma_z'] = sigma_z
    data['Phi_corr'] = Phi_corr
    data['Phi_cov'] = Phi_cov
    data['z'] = zz
    data['y'] = yy
    data['D'] = DD
    data['off_diag_residual'] = off_diag_residual
    data['cross_loadings'] = cross_loadings
    data['method'] = method
    data['model_type'] = model_type

    try:
        data['Theta'] = Theta
    except:
        pass
    try:
        data['Theta_corr'] = Theta_corr
    except:
        pass
    try:
        data['sigma_theta'] = sigma_theta
    except:
        pass
    try:
        data['e'] = ee
    except:
        pass
    try:
        data['c'] = c
    except:
        pass

    return(data)


def gen_data_binary_1factor(nsim_data, J=6, K=1, c=1, noise=False,
                            cheaters=False, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    # alpha = np.array([-0.53,  0.35, -1.4 , -1.4 , -0.96, -2.33])
    alpha = np.zeros(J)
    beta = np.array([1, 0.7, .8, .5, .9, .6])

    zz = norm.rvs(scale=c, size=nsim_data)
    yy = alpha + np.outer(zz, beta)
    if cheaters:  # add cheaters
        yy[900:, 3:] = yy[900:, 3:] + 0.5

    DD = bernoulli.rvs(p=expit(yy))

    if noise:  # replace noisy column
        noisy_col = bernoulli.rvs(p=0.5, size=nsim_data)
        DD[:, 0] = noisy_col

    data = dict()
    data['noise'] = noise
    data['cheaters'] = cheaters
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['sigma'] = c
    data['z'] = zz
    data['y'] = yy
    data['D'] = DD

    return(data)


def check_posdef(R):
    """
    Check that matrix is positive definite
    Inputs
    ============
    - matrix
    """
    try:
        cholesky(R)
    except NameError:
        print("\n Error: could not compute cholesky factor of R")
        raise
        sys.exit(1)
    return 0


def extract_values_above_diagonal(M, offset=1):
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
    return M[np.triu_indices(n=M.shape[0], k=offset)]


def flatten_df(df0, val_name, var_name='K'):
    df = pd.DataFrame(df0)
    df.columns = df.columns + 1
    df['J'] = np.arange(len(df))+1
    return df.melt(id_vars=['J'], var_name=var_name, value_name=val_name)


def post_summary(samples):
    mean = pd.DataFrame(np.mean(samples, axis=0))
    ps_df = flatten_df(mean, 'mean')
    median = pd.DataFrame(np.median(samples, axis=0))
    ps_df['median'] = flatten_df(median, 'median')['median']
    per1 = pd.DataFrame(np.percentile(samples, 2.5, axis=0))
    ps_df['q2.5'] = flatten_df(per1, 'q2.5')['q2.5']
    per2 = pd.DataFrame(np.percentile(samples, 97.5, axis=0))
    ps_df['q97.5'] = flatten_df(per2, 'q97.5')['q97.5']
    return ps_df


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

    N, K = Rs.shape[0], Rs.shape[1]
    if colnames is None:
        colnames = [str(x) for x in range(K)]

    assert len(colnames) == K, 'colnames should as long as the columns of R'
    cnames = corr_headers(colnames)

    M = len(cnames)
    k = 0
    fRs = np.empty((N, M))
    for i in range(N):
        fRs[i, :] = flatten_corr(Rs[i, :, :])

    fRs = pd.DataFrame(fRs)
    fRs.columns = cnames

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
    dfcols = list(zip(colnames[np.triu_indices(colnames.shape[0], k=1)[0]],
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
