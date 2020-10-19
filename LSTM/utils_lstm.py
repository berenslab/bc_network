import numpy as np
import scipy as scp

import torch

from random_search import  randint, loguniform


def get_sample():
    config = {
        'hash': randint(1000000, 9999999),
        'lr': loguniform(1e-1, 5e0),
        'max_steps': randint(10000, 50000),
        'decrease_lr_after': randint(20, 80),
        'stop_after': randint(10, 20),
        'seed': randint(1000000, 9999999),
        }
    return config


def correlation(x, y):
    """
    returns correlation, for normalized inputs.
    ---
    shape (tpts, ntraces)
    """
    return torch.mean(torch.sum(x * y, dim=0))

def normalize(outputs):
    """
    torch tensors, shape (tpts, ntraces)
    """
    outputs = outputs - outputs.mean(dim=0)
    outputs = outputs / torch.norm(outputs, dim=0)
    # if norm==0 it results in nans. replace here:
    outputs[torch.isnan(outputs)] = 0
    return outputs

def mse(x,y):
    """
    numpy
    """
    return np.mean((x-y)**2)

def comput_nr_params(model):
    param_count = 0 
    for i in range(len(list(model.parameters()))):
        if len(list(model.parameters())[i].size()) ==2:
            param_count += (list(model.parameters())[i].size())[0] * list(model.parameters())[i].size()[1]
        else:
            param_count += (list(model.parameters())[i].size())[0]
    return param_count