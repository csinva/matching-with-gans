import numpy as np
import numpy.linalg as npl
from copy import deepcopy
def orthogonalize_paper(vs: np.ndarray):
    '''
    Params
    ------
    vs
        matrix of all vectors to orthogonalize
        (each col is a vector)
    '''
    d = vs.shape[1] # number of vectors
    Q, R = npl.qr(vs)
    vs_orth = vs.copy()
    for i in range(d):
        for j in range(d):
            if not i == j:
                scalar = np.dot(Q[:, j], vs_orth[:, i]) / npl.norm(Q[:, j])
                vs_orth[:, i] -= Q[:, j] * scalar
        vs_orth[:, i] = vs_orth[:, i] / npl.norm(vs_orth[:, i])
    return vs_orth
#     for i in range

def orthogonalize(v0: np.ndarray, vs: np.ndarray):
    '''
    Params
    ------
    v0: (size)
        vector to orthogonalize
    vs: (num_vectors x size)
        vectors to orthogonalize against
    '''
    '''
    # convert other vectors to a matrix
    A = np.zeros((max(v0.shape), len(vs)))
    for i, h in enumerate(vs):
        A[:, i] = h.coef_
    '''
    
    # decompose the matrix
    vs = vs.transpose() # make it (size x num_vectors)
    Q, R = npl.qr(vs)

    # subtract projections onto other vectors
    u = v0.copy()
    for i in range(vs.shape[1]):
        u -= proj(Q[:, i], u)
    
    # normalize
    u = u / npl.norm(u)
    return u

def proj(u: np.ndarray, v: np.ndarray):
    '''Return projection of u onto v
    '''
    return u * np.dot(u, v) / np.dot(u, u)