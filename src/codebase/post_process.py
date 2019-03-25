import numpy as np
import operator
from numpy.linalg import eigh

def get_topn(L, topn):
    """
    Return the top n values a list L, along with their indices
    """
    index = np.empty(topn, dtype=int)
    max_evals = np.empty(topn)
    L_c = L.copy()
    for i in range(topn):
        # find highest element of L_c, with its index
        ind, value = max(enumerate(L_c), key=operator.itemgetter(1))
        index[i] = ind
        max_evals[i] = value
        # delete element to repeat process
        L_c = np.delete(L_c,ind)
    return index, max_evals

def get_topn_eig(M, topn):
    eset = eigh(M)
    L = eset[0]
    P = eset[1]
    index, max_evals = get_topn(L, topn)


    out = dict()
    out['index'] = index
    out['P'] = P[:, index]
    for i in range(topn):
        if  out['P'][0,i]<0:
            out['P'][:,i] = -out['P'][:,i]

    out['L'] = L[index]

    return out


def get_non_zeros(x, prc_min = 10, prc_max = 90):
    """
    Returns the index of the elements that do not contain
    zero in their quantile interval from prc_min to prc_max.
    Indices are returned in two arrays a1, a2. The i-th non
    zero element of the matrix x is at position [a1[i], a2[i]].
    """
    rcs = np.percentile(x, [prc_min,prc_max], axis=0)
    min_b = rcs[0,:,:]
    max_b = rcs[1,:,:]

    zeros = np.zeros((x.shape[1], x.shape[2]))

    indx = (min_b < zeros) & (zeros < max_b)
    # data['u'][~indx]
    return np.nonzero(~indx)
