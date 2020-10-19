import torch
import numpy as np
import pickle
from scipy import interpolate
import os
import pandas as pd


class WeightInitialization:
    def __init__(
            self,
            shape=None,
            noise_init_scale=1e-2,
            scale='log',  # lin (-inf, inf), log (0, inf), sig (0, 1)
            requires_grad=True,
            random=True,
            seed=1234,
    ):
        '''
        class for weight initialization.
        :param num_cells: how many cell types (to expand float inits).
        :param noise_init_scale: standard deviation of noise initialization.
        :param log: boolean. return parameter in log scale
        :param requires_grad: whether to fit.
        :param random: if False: no random initialization takes place.
        '''
        self.shape = shape
        self.noise_init_scale = noise_init_scale
        self.scale = scale
        self.requires_grad = requires_grad
        self.random = random
        self.seed = seed

    def initialize(self, init, shape=None, scale=None, requires_grad=None,
                   random=None):
        """
        for weight initialization. scales noise up if init>1.
        :param init: initializing value, tensor, np.array or float
        :param scale: which scale to use
        :param fit: if False: no random initialization takes place and no grad.
        :return: noisy initial values
        """
        if shape is None:
            shape = self.shape
        if scale is None:
            scale = self.scale
        if requires_grad is None:
            requires_grad = self.requires_grad
        if random is None:
            random = self.random
        init = torch.tensor(init, dtype=torch.float32)
        if not init.shape:
            init = init * torch.ones(shape)
        if random:
            torch.manual_seed(self.seed)
            noise = torch.randn(init.shape) * self.noise_init_scale
            # if largest absolute value is > 1, scale up noise
            noise *= 1 + torch.relu(torch.max(torch.abs(init)) - 1)
            init += noise
        if scale == 'lin':
            pass
        elif scale == 'log':
            init = torch.log(init.clamp(min=1e-6))
        elif scale == 'sig':
            init = init.clamp(min=1e-6, max=1 - 1e-6)
            init = torch.log(init / (1 - init))
        self.seed += 1  # change seed for next call
        return torch.nn.Parameter(data=init, requires_grad=requires_grad)


def get_iGluSnFR_kernel(
        duration=0.3,  # how long the kernel will be
        dt=1 / 30,  # One over sampling frequency
        tau_r=-0.09919711,  # rise time constant
        tau_d=-0.04098927):  # decay time constant
    # from https://github.com/berenslab/abc-ribbon/blob/master/standalone_model/ribbonv2.py
    t_kernel = np.arange(0, duration, dt)
    kernel = np.exp(t_kernel / tau_d) - np.exp(
        (tau_d + tau_r) / (tau_d * tau_r) * t_kernel)
    kernel /= np.sum(kernel)
    kernel = kernel.astype(np.float32)[None, None, ::-1]  # out x in x depth
    return kernel.copy()


def get_new_chirp(data_dir="data/"):
    """
    loading resampling data to new frequency
    ---
    f_in: frequency of input
    f_out: desired frequency
    """
    file = os.path.join(
        data_dir, 'StimulusFranke2017time_and_amp_corrected_SIGMOID.csv'
    )
    stim_new = pd.read_csv(file)
    data = stim_new['Stim']
    t_in = np.array(stim_new['Time'])
    f_in = 1000
    f_out = 64
    t_out = np.arange(0, t_in[-1], 1 / f_out)
    data_out = interpolate.interp1d(t_in, data)(t_out)
    return data_out


def get_data(
        normalize=True,
        data_dir="data/",
        chirp='new_jonathan',
):
    file = os.path.join(
        data_dir, "Franke_etal_clustermeans_no_drugs.pkl"
    )
    with open(file, "rb") as f:
        dictname = pickle.load(f)
    local_chirp = dictname['local_chirp']
    global_chirp = dictname['global_chirp']
    idxs = []
    for key in local_chirp:
        idxs.append(key)
    types = {'NO': None,
             '1': 'OFF',
             '2': 'OFF',
             '3a': 'OFF',
             '3b': 'OFF',
             '4': 'OFF',
             '5t': 'ON',
             '5o': 'ON',
             '5i': 'ON',
             'X': 'ON',
             '6': 'ON',
             '7': 'ON',
             '8': 'ON',
             '9': 'ON',
             'R': 'rod'}
    # dt of recordings (different of dt_stim!!!)
    dt = np.diff(global_chirp['BC1']['Time'])[0]
    sampling_frequency = 1 / dt
    exp_time = global_chirp['BC1']['Time']  # experimental time
    # construct data matrix
    X_local = np.zeros((14, 2048))
    X_global = np.zeros((14, 2048))
    local_m, local_norm = np.zeros(14), np.zeros(14)
    global_m, global_norm = np.zeros(14), np.zeros(14)
    for i in range(14):
        X_local[i] = local_chirp[idxs[i + 1]]['mean']
        X_global[i] = global_chirp[idxs[i + 1]]['mean']
        if normalize:
            local_m[i] = np.mean(X_local[i])
            X_local[i] -= local_m[i]
            local_norm[i] = np.linalg.norm(X_local[i])
            X_local[i] /= local_norm[i]

            global_m[i] = np.mean(X_global[i])
            X_global[i] -= global_m[i]
            global_norm[i] = np.linalg.norm(X_global[i])
            X_global[i] /= global_norm[i]
        # print(i)
    # load  stimulus
    if chirp == 'new_jonathan':
        chirp_file = os.path.join(
            data_dir, 'StimulusFranke2017time_and_amp_corrected_SIGMOID.csv'
        )
        chirp_data = pd.read_csv(chirp_file)
        f = interpolate.interp1d(chirp_data['Time'], chirp_data['Stim'])
    elif chirp == 'jonathan':
        stim = dictname['stim']
        light = stim.Stim.values
        light_time = stim.Time.values
        f = interpolate.interp1d(light_time, light)
    elif chirp == 'old':
        stim = np.load('data/old_chirp.npy')
        f = interpolate.interp1d(np.linspace(0, 32, stim.shape[0]), stim)
    # downsample  stimulus
    xnew = np.arange(0, 32, 1 / 64)
    chirp = f(xnew)  # use interpolation function returned by `interp1d`
    chirp -= np.mean(chirp)
    chirp /= np.std(chirp)
    return (X_local, X_global, chirp, types, sampling_frequency, exp_time,
            local_m, local_norm, global_m, global_norm)


def precomputed_sigmoid(x, sigmoid_offset, sigmoid_slope):
    """
    Same as sigmoid of the LNR model, but precomputed to be faster in loop.
    :param x: C
    :param: sigmoid_offset
    :param sigmoid_slope
    :return:
    """
    return torch.sigmoid((x - sigmoid_offset) * sigmoid_slope)


def torch_correlation(x, y):
    """
    returns correlation, for normalized inputs.
    """
    return torch.mean(torch.sum(x * y, dim=1))
