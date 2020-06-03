import torch
import numpy as np
import numpy.linalg as npl
from copy import deepcopy
import scipy.stats

def detach(tensor):
    return tensor.detach().cpu().numpy()

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

def spearman_mean(y_pred: torch.Tensor, y_train: torch.Tensor):
    '''
    Params
    ------
    y_pred
        (n_samples, n_attributes)
    y_train
        (n_samples, n_attributes)
        
    Returns
    -------
    mean_rho: float
        mean spearman correlation between corresponding columns
        of y_pred and y_train
    '''
    
    
    spearman_cum = 0
    for i in range(y_pred.shape[1]):
        spearman_cum += scipy.stats.spearmanr(detach(y_pred[:, i]),
                                              detach(y_train[:, i])).correlation
    return spearman_cum / y_pred.shape[1]


def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    xm = x - torch.mean(x)
    ym = y - torch.mean(y)
    return xm.dot(ym) / (torch.norm(xm, 2) * torch.norm(ym, 2))


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