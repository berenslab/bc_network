import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse
import json
import traceback
import shutil
import sys
import pickle
import pickle as pkl
from model_ln import LinearNonlinear
from random_search import  randint, loguniform
from utils import get_data


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

def get_sample():
    config = {
        'hash': randint(1000000, 9999999),
        'lr': loguniform(1e-2, 1e0),
        'max_steps': randint(3000, 10000),
        #'time_reg_weight': loguniform(3e-4, 3e0),
        #'sparsity_reg_weight': loguniform(1e-6, 1e-2),
        'decrease_lr_after': randint(10, 50),
        'stop_after': randint(10, 20),
        'seed': randint(1000000, 9999999),
        'noise_init_scale': loguniform(1e-3, 5e-1)
    }
    return config

def train(
        model,
        stimulus,
        response,
        log_dir='.',  # where to save the model
        log_file=None,
        log_interval=5,
        lr=1e-1,  # learning rate
        time_reg_weight=0,  #1e1,  # 3e-2
        max_steps=500,
        decrease_lr_after=3,  # if loss went up x times, lower lr
        stop_after=5,  # if lr was lowered x times, stop training
        verbose=False
):
    if log_file is None:
        log_file = open(os.path.join(log_dir, "train_log.txt"), 'a', 1)
        close_when_done = True
    else:
        close_when_done = False

    print('Running Model:', model, file=log_file)
    print('lr=%s, max_steps=%s,' % (
        lr, max_steps), file=log_file)
    params_before = []
    for n in model.named_parameters():
        if n[1].requires_grad:
            if 'log_' in n[0]:
                param = np.exp(n[1].cpu().detach().numpy())
                name = n[0].replace('log_', '')
            else:
                param = n[1].cpu().detach().numpy()
                name = n[0]
            params_before.append(param)
            #print(name + '=%s\n' % (params_before[-1].flatten()),
            #      file=log_file)

    device = torch.device("cuda:0")
    model = model.to(device)

    # define stimulus
    x = torch.tensor(stimulus.astype(np.float32)).to(device)

    # define response we want to fit on
    y = torch.tensor(response.astype(np.float32)).to(device)
    y =  normalize(y.T)

    #criterion = nn.MSELoss()
    criterion = correlation

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    t0 = time.time()
    t_start = time.time()

    track_loss=[]
    running_loss = np.zeros(3)  # total, trace, kernel_reg,
    best_mse = np.inf
    not_improved = 0
    stop_count = 0

    # Training Loop
    for i in range(max_steps + 1):
        optimizer.zero_grad()
        y_= model.forward(x)

        # loss
        y_ = normalize(y_)
        loss = - criterion(y, y_)
        running_loss[1] += loss.item()
        track_loss.append(loss.item())

        # regularize L2 of log_speeds to be near 0 (i.e. speed=1)
        '''

        reg_speed = time_reg_weight * torch.mean(
                model.log_kernel_speed**2)

        running_loss[2] += reg_speed.item()
        loss = loss + reg_speed
        '''
        '''
        # L1 penalty on connectivity
        reg_sparsity = sparsity_reg_weight * (
            torch.mean(torch.exp(model.log_ac_bc_weight)) +
            torch.mean(torch.exp(model.amacrine_cells.log_bc_ac_weight)))
        running_loss[3] += reg_sparsity.item()
        loss = loss + loss_global + reg_sparsity
        '''

        running_loss[0] += loss.item()

        # always track
        if not (i % log_interval) and i > 0:
            running_loss /= log_interval
            print('Step %s (%.2fs), Loss: Total=%s, Trace=%s, Speed_reg=%.2e, ' % (
                      i, time.time() - t0, *running_loss), file=log_file)
            if verbose:
                print('Step %s (%.2fs), Loss: Total=%s, Trace=%s, Speed_reg=%.2e, ' % (
                    i, time.time() - t0, *running_loss))

            t0 = time.time()
            running_loss = np.zeros(3)


        # early stopping
        if loss.item() > best_mse:
            not_improved += 1
            if not_improved >= decrease_lr_after:
                # go back to last best
                model.load_state_dict(
                    torch.load(os.path.join(log_dir, 'model.pth')))
                stop_count += 1
                if stop_count >= stop_after:
                    print('Converged.', file=log_file)
                    if verbose:
                        print('current loss:', loss.item())
                        print('Converged.')
                    break
                lr *= .5
                for g in optimizer.param_groups:  # decrease lr
                    g['lr'] *= .5
                print('lowering learning rate to ', lr, file=log_file)
                if verbose:
                    print('lowering learning rate to ', lr)
                not_improved = 0
            else:
                # make step if it continues at this lr
                loss.backward()
                optimizer.step()
        else:
            torch.save(
                model.state_dict(), os.path.join(log_dir, 'model.pth'))
            not_improved = 0
            best_mse = loss.item()
            # make step after saving best
            loss.backward()
            optimizer.step()

    # when done
    if stop_count < stop_after:
        print('Maximum number of steps reached.', file=log_file)
        if verbose:
            print('Maximum number of steps reached.')

    if close_when_done:
        log_file.close()

    data_dict = {'best_loss': best_mse,
                'track_loss': track_loss,
                #'track_lr_change': track_lr_change,
                #'track_lr': track_lr,
                #'track_best_indeces': track_best_indeces,
                #'traces': outputs.cpu().detach(),
                'runtime': time.time()-t_start}

    with open(log_dir+'/loss_tracking.pkl', "wb") as f:
        pkl.dump(data_dict,f)
#############################################################################################

if __name__ == '__main__':

    """
    training
    random search
    """
    data_mode = 'local'  # in ['local', 'global']

    random_search = True
    n_runs = 100

    log_file = None

    args = {
        'log_dir': 'temp',
        'log_interval': 100}

    args['log_dir'] = args['log_dir'] + data_mode + '/'

    # define cell types
    cell_types = np.concatenate([np.ones(5) * -1, np.ones(9)])

    # load data
    (y_local, y_global, chirp, types, sampling_frequency, exp_time,
     local_m, local_sd, global_m, global_sd) = get_data(
        normalize=True)  # args.normalize

    # set stim and response
    stimulus = chirp.astype(np.float32)

    if data_mode == 'local':
        response = y_local.astype(np.float32)
    elif data_mode == 'global':
        response = y_global.astype(np.float32)

    # random search
    if random_search:
        # ToDO: add seed
        # while True:
        for i in range(n_runs):
            config = get_sample()
            model = LinearNonlinear(cell_types,
                                    noise_init_scale=config['noise_init_scale'],
                                    seed=config['seed'])

            if config['hash'] in os.listdir(args['log_dir']):
                while True:
                    config['hash'] += 1
                    if config['hash'] not in os.listdir(args['log_dir']):
                        break
            log_dir = os.path.join(args['log_dir'], str(config['hash']))
            os.mkdir(log_dir)
            train(
                model,
                stimulus,
                response,
                log_dir=log_dir,
                log_interval=args['log_interval'],
                # time_reg_weight=config['time_reg_weight'],
                lr=config['lr'],
                max_steps=config['max_steps'],
                decrease_lr_after=config['decrease_lr_after'],
                stop_after=config['stop_after'],
                verbose=False
            )
            # and save configs:
            with open(log_dir + '/config.pkl', "wb") as f:
                pkl.dump(config, f)


    else:
        config = get_sample()
        model = LinearNonlinear(cell_types,
                                noise_init_scale=config['noise_init_scale'],
                                seed=config['seed'])
        if config['hash'] in os.listdir(args['log_dir']):
            while True:
                config['hash'] += 1
                if config['hash'] not in os.listdir(args['log_dir']):
                    break
        log_dir = os.path.join(args['log_dir'], str(config['hash']))
        os.mkdir(log_dir)

        train(
            model,
            stimulus,
            response,
            log_dir=log_dir,
            log_interval=args['log_interval'],
            # time_reg_weight=config['time_reg_weight'],
            lr=config['lr'],
            max_steps=config['max_steps'],
            decrease_lr_after=config['decrease_lr_after'],
            stop_after=config['stop_after'],
            verbose=False
        )
        with open(log_dir + '/config.pkl', "wb") as f:
            pkl.dump(config, f)