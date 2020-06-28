import torch

import util


def calc_mean_corrs_between_attributes(Ahat: torch.Tensor):
    '''Calculate mean correlation off-diagonal
    Ahat: torch.Tensor
        attributes (num_samples, num_attributes)
    '''
    num = Ahat.shape[1]
    corr = 0
    for i in range(num):
        for j in range(i):
            corr += torch.abs(util.pearsonr(Ahat[:, i], Ahat[:, j]))
    return corr / ((num * (num - 1)) / 2)

def mse(y_pred, y_true):
    return torch.mean(torch.square(y_pred - y_true))
