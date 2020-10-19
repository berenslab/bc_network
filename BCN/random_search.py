import numpy as np


def loguniform(low, high):
    return np.exp(
        np.random.uniform(np.log(low), np.log(high), 1)
    )[0]


def randchoice(options):
    return np.random.choice(options, 1)[0]


def randint(low, high):
    return np.int(np.random.randint(low, high, 1)[0])


def randbool():
    return np.random.choice([True, False], 1)[0]


def get_sample():
    noise_scale = randbool()
    if noise_scale:
        noise_scale = loguniform(1e-1, 1e1)  # multiplied by lr
    else:
        noise_scale = 0
    config = {
        'hash': randint(1000000, 9999999),
        'lr': loguniform(1e-2, 1e0),
        'max_steps': 1000,  # randint(100, 2000),
        'time_reg_weight': loguniform(1e-2, 1e-1),
        'sparsity_reg_weight': loguniform(1e-8, 1e0),
        'scaling_mean_weight': loguniform(1e-3, 1e1),
        'scaling_std_weight': loguniform(1e-3, 1e1),
        'decrease_lr_after': randint(5, 100),
        'stop_after': randint(3, 10),
        'noise_init_scale': loguniform(1e-4, 1e0),
        'load_best_lnr': bool(randbool()),
        'noise_scale': noise_scale,
        'load_best_previous': bool(randbool()),
    }
    return config