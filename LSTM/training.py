import numpy as np
import time
import torch
import torch.nn as nn
import pickle as pkl
# get data
import sys 
sys.path.append("..")
from utils import get_data
import os
# get utils
from utils_lstm import correlation, normalize, mse, comput_nr_params
# get model
from model import LSTMModel


"""
define trainig
"""
def train(model, #lstm model
             data,
             stimulus,
             loss_mode = 'corr', # in ['corr', 'mse']
             lr = 5e-1,
             decrease_lr_after=20, #10  # if loss went up x times, lower lr # corr: 20, mse:10
             stop_after= 20,  # if lr was lowered x times, stop training # corr: 20, mse:10
             log_file = None,
             max_steps = 20000,
             log_dir = None, # at the moment: os.getcwd()+'/results/models/'
             device = 'cuda:0',
             verbose = True):
    
    
    #log_dir = os.getcwd()+'/results/models/'
    t0 = time.time()
    # to GPU    
    model.to(device)
    
    # Number of steps to unroll
    seq_dim = stimulus.shape[0]

    # define stimulus
    stimulus = torch.tensor(stimulus, dtype=torch.float32)
    stimulus = stimulus.view(-1, seq_dim, model.input_dim).requires_grad_().to(device)

    # define response
    response = torch.tensor(data.T, dtype=torch.float32)
    # normalize to use correlation
    response = normalize(response).to(device)

    #define loss mode
    if loss_mode=='corr':
        criterion = correlation
    elif loss_mode=='mse':
        criterion = nn.MSELoss()

    # initialize tracking
    track_loss = []
    track_lr_change = []
    track_lr = []
    track_best_indeces = []
    track_lr.append(lr)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    
    # setting initial parameters
    best_loss = np.inf
    not_improved = 0
    stop_count = 0

    if log_file is None:
        log_file = open(os.path.join(log_dir, "train_log.txt"), 'a', 1)
        close_when_done = True
    else:
        close_when_done = False
        
    # gradient descent loop
    for i in range(max_steps + 1):
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(stimulus)[0] # only one batch

        # normalize 
        outputs = normalize(outputs)

        if loss_mode=='corr':    
            loss = - criterion(outputs, response)
        elif loss_mode=='mse':
            loss = criterion(outputs, response)

        track_loss.append(loss.cpu().detach())

        # early stopping
        if loss.item() > best_loss: 
            not_improved += 1
            if not_improved >= decrease_lr_after:
                # go back to last best
                model.load_state_dict(
                    torch.load(os.path.join(log_dir, 'model.pth')))
                stop_count += 1
                if stop_count >= stop_after:
                    print('Converged.', file=log_file)
                    #print('Converged.')
                    break
                lr *= .5
                track_lr_change.append(i)
                track_lr.append(lr)
                for g in optimizer.param_groups:  # decrease lr
                    g['lr'] *= .5
                print('iteration: {}. lowering learning rate to {}. '.format(i, lr), file=log_file)
                if verbose:
                    print('iteration: {}. lowering learning rate to {}. '.format(i, lr))
                not_improved = 0
            else:
                # make step if it continues at this lr
                loss.backward()
                optimizer.step()
        else:
            torch.save(
                model.state_dict(), os.path.join(log_dir, 'model.pth'))
            best_loss = loss.item()
            track_best_indeces.append(i)
            # make step after saving best
            loss.backward()
            optimizer.step()
            not_improved=0

        # printing to file/notebook
        if i % 100 == 0:
            # Print Loss
            print('Iteration: {}. mean Loss of last 100 steps: {}.'.format(i, np.mean(track_loss[i-100:i]))
                 , file=log_file)
            if verbose:
                print('Iteration: {}. mean Loss of last 100 steps: {}.'.format(i, np.mean(track_loss[i-100:i])))
            if verbose:
                print('  .... current loss: {}'.format(loss.item()))
            print('  .... current loss: {}'.format(loss.item()), file=log_file)

        if i==max_steps:
            print('max steps reached.', file=log_file)
            if verbose:
                print('max steps reached.')

    if close_when_done:
        log_file.close()
        
    # final fit
    outputs = model(stimulus)[0] # only one batch
    # normalize 
    outputs = normalize(outputs)
    
    
    data_dict = {'best_loss': best_loss, 
                'track_loss': track_loss, 
                'track_lr_change': track_lr_change,
                'track_lr': track_lr,
                'track_best_indeces': track_best_indeces,
                'traces': outputs.cpu().detach(),
                'runtime': time.time()-t0}

    with open(log_dir+'/loss_tracking.pkl', "wb") as f:
        pkl.dump(data_dict,f)
        

if __name__== 'main':
    """
    load data
    returns: (X_local, X_global, chirp, types, sampling_frequency, exp_time,
                local_m, local_sd, global_m, global_sd)
    """
    X_local, X_global, chirp, types, sampling_frequency, *_  = get_data(normalize=True)
    
