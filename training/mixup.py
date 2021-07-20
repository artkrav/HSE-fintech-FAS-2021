import torch
import numpy as np


def get_perm(x, use_cuda=True):
    """get random permutation"""
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    return index


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def class_dominant_mixup(data, label, lamb=0.7):
    live_idx = label > 0.5
    spoof_idx = label <= 0.5
    num_min = torch.min(live_idx.sum(), spoof_idx.sum())
    
    live_data, live_label = data[[live_idx]][:num_min], label[[live_idx]][:num_min]
    spoof_data, spoof_label = data[[spoof_idx]][:num_min], label[[spoof_idx]][:num_min]
    
    mixed_ld = lamb * live_data + (1 - lamb) * spoof_data
    mixed_sd = lamb * spoof_data + (1 - lamb) * live_data
    return mixed_ld, live_label, mixed_sd, spoof_label