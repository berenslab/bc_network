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
from models import LinearNonlinearRelease, BipolarCellNetwork, FullBCModel
# lnr, bcn, full
from utils import get_data, torch_correlation
from random_search import get_sample


def train(
        model,
        stimulus,
        responses,
        log_dir='.',  # where to save the model
        log_file=None,
        log_interval=5,
        time_reg_weight=1e1,  # 3e-2
        sparsity_reg_weight=1e-3,  # for BCN model (before: 1e-3)
        scaling_mean_weight=1e0,  # to keep release means similar
        scaling_std_weight=1e0,  # to keep release std.dev.s similar
        lr=1e-1,  # learning rate
        noise_scale=0,
        max_steps=500,
        decrease_lr_after=3,  # if loss went up x times, lower lr
        stop_after=5,  # if lr was lowered x times, stop training
):
    if log_file is None:
        log_file = open(os.path.join(log_dir, "train_log.txt"), 'a', 1)
        close_when_done = True
    else:
        close_when_done = False

    # other logs
    log_general = open(os.path.join(log_dir, 'log_general.csv'), 'a', 1)
    log_general.write('loss_local,loss_global,reg_speed,reg_sparsity,'
                      'reg_scale_mean,reg_scale_std,total,lr\n')
    log_local = open(os.path.join(log_dir, 'log_local.csv'), 'a', 1)
    log_global = open(os.path.join(log_dir, 'log_global.csv'), 'a', 1)

    print('Running Model:', model, file=log_file)
    print('lr=%s, max_steps=%s, time_reg_weight=%s, sparsity_reg_weight=%s, '
          'scale_mean_weight=%s, scale_std_weight=%s\nInit Params:\n' % (
        lr, max_steps, time_reg_weight, sparsity_reg_weight,
        scaling_mean_weight, scaling_std_weight), file=log_file)
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

    x = torch.tensor(stimulus.astype(np.float32)).to(device)
    y_local = torch.tensor(responses[0].astype(np.float32)).to(device)
    y_global = torch.tensor(responses[1].astype(np.float32)).to(device)

    # criterion = nn.MSELoss()
    criterion = torch_correlation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    t0 = time.time()
    running_loss = np.zeros(7)
    # Local, Global, Speed, Sparsity, Scaling mean, Scaling std, Total
    lowest_loss = np.inf
    not_improved = 0
    stop_count = 0

    # Training Loop
    for i in range(max_steps + 1):
        optimizer.zero_grad()
        y_local_, lnr_state, y_global_, bcn_state = model.forward(x)

        # Local
        loss_local = 1 - criterion(
            y_local_, y_local)  # + torch.randn_like(y_local) * lr / 10
        running_loss[0] += loss_local.item()
        # regularize std.dev. of log_speeds
        reg_speed = time_reg_weight * torch.std(
            torch.exp(model.log_kernel_speed))
        running_loss[2] += reg_speed.item()
        loss = loss_local + reg_speed

        # Global
        loss_global = 1 - criterion(
            y_global_, y_global)  # + torch.randn_like(y_global) * lr / 10, )
        running_loss[1] += loss_global.item()
        # L1 penalty on connectivity
        reg_sparsity = sparsity_reg_weight * (
            torch.sum(torch.exp(model.log_acl_bc_weight)) +
            torch.sum(torch.exp(
                model.glycinergic_amacrine_cells.log_bc_ac_weight)) +
            torch.sum(torch.exp(model.log_acg_bc_weight)) +
            torch.sum(torch.exp(
                model.gabaergic_amacrine_cells.log_bc_ac_weight)) +
            torch.sum(torch.exp(model.log_acl_acg_weight))
        )
        running_loss[3] += reg_sparsity.item()
        # scaling penalty
        releases = torch.cat([  # time x channel
            lnr_state['track_release'], bcn_state['track_release']], dim=1)
        means = torch.mean(releases, dim=0)
        stds = torch.std(releases, dim=0)
        scaling_mean_penalty = scaling_mean_weight * torch.std(means)
        scaling_std_penalty = scaling_std_weight * torch.std(stds)
        running_loss[4] += reg_sparsity.item()
        running_loss[5] += reg_sparsity.item()

        # final loss
        loss = loss + loss_global + reg_sparsity + scaling_mean_penalty + \
               scaling_std_penalty

        if torch.isnan(loss):
            print('cancel because of nan')
            return False

        running_loss[6] += loss.item()

        # always track
        log_general.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (
            loss_local.item(), loss_global.item(),
            reg_speed.item(), reg_sparsity.item(),
            loss.item(), scaling_mean_penalty.item(),
            scaling_std_penalty.item(), lr))
        log_local.write(','.join(
            ['%s' % torch.sum(a * b).item() for a, b in zip(
                y_local, y_local_)]) + '\n')
        log_global.write(','.join(
            ['%s' % torch.sum(a * b).item() for a, b in zip(
                y_global, y_global_)]) + '\n')

        if not (i % log_interval) and i > 0:
            running_loss /= log_interval
            print('Step %s (%.2fs), Loss: Local=%s, Global=%s, '
                  'Speed_reg=%.2e, Sparsity_reg=%.2e, Scaling_mean_reg=%.2e,'
                  ' Scaling_std_reg=%.2e, Total=%.2e' % (
                      i, time.time() - t0, *running_loss), file=log_file)
            t0 = time.time()
            running_loss = np.zeros(7)

        # early stopping
        if loss.item() > lowest_loss:
            not_improved += 1
            if not_improved >= decrease_lr_after:
                # go back to last best
                model.load_state_dict(
                    torch.load(os.path.join(log_dir, 'model.pth')))
                stop_count += 1
                if stop_count >= stop_after:
                    print('Converged.', file=log_file)
                    break
                lr *= .5
                for g in optimizer.param_groups:  # decrease lr
                    g['lr'] *= .5
                print('lowering learning rate to ', lr, file=log_file)
                not_improved = 0
            else:
                # make step if it continues at this lr
                loss.backward()
                optimizer.step()
        else:
            torch.save(
                model.state_dict(), os.path.join(log_dir, 'model.pth'))
            lowest_loss = loss.item()
            # make step after saving best
            loss.backward()
            optimizer.step()

        for p in model.parameters():
            s = lr * noise_scale
            if p.requires_grad:
                if p.shape[0] > 1:
                    p.data += torch.randn_like(p.data) * s * torch.std(p.data)
                else:
                    p.data += torch.randn_like(p.data) * s


    # when done
    if stop_count < stop_after:
        print('Maximum number of steps reached.', file=log_file)

    ind = 0
    for n in model.named_parameters():
        if n[1].requires_grad:
            if 'log_' in n[0]:
                param = np.exp(n[1].cpu().detach().numpy())
                name = n[0].replace('log_', '')
            else:
                param = n[1].cpu().detach().numpy()
                name = n[0]
            print(name, 'before=%s\nafter=%s\n' % (
                params_before[ind].flatten(), param.flatten()), file=log_file)
            ind += 1

    if close_when_done:
        log_file.close()
    return True


def main(args, log_file=None):
    cell_types = np.concatenate([np.ones(5) * -1, np.ones(9)])
    (y_local, y_global, chirp, types, sampling_frequency, exp_time,
     local_m, local_norm, global_m, global_norm) = get_data(
        chirp=args.chirp
    )

    def model_init(load_best_lnr, seed=args.seed,
                   noise_init_scale=args.noise_init_scale):
        if load_best_lnr:
            with open('results/best_model/best_parameters', 'rb') as f:
                bestparams_dict = pickle.load(f)
            return FullBCModel(
                cell_types=cell_types,
                change_prob01_init=bestparams_dict['change_prob01'],
                change_prob12_init=bestparams_dict['change_prob12'],
                intermediate_pool_capacity_init=bestparams_dict[
                    'intermediate_pool_capacity'],
                release_pool_capacity_init=bestparams_dict[
                    'release_pool_capacity'],
                sigmoid_offset_init=bestparams_dict['sigmoid_offset'],
                sigmoid_slope_init=bestparams_dict['sigmoid_slope'],
                kernel_speed_init=bestparams_dict['kernel_speed'],
                noise_init_scale=noise_init_scale,
                fit_release_parameters=not args.fix_release_parameters,
                fit_linear_filter=not args.fix_linear_filter,
                fit_non_linearity=not args.fix_non_linearity,
                fit_steady_state=not args.fix_steady_state,
                random_init=not args.no_random_init,
                seed=seed,
                ip_steady=bestparams_dict['ip_steady'],
                rrp_steady=bestparams_dict['rrp_steady'],
                kernel_type=args.kernel_type,
            )
        else:
            return FullBCModel(
                cell_types=cell_types,
                noise_init_scale=noise_init_scale,
                fit_release_parameters=not args.fix_release_parameters,
                fit_linear_filter=not args.fix_linear_filter,
                fit_non_linearity=not args.fix_non_linearity,
                fit_steady_state=not args.fix_steady_state,
                random_init=not args.no_random_init,
                seed=seed,
                kernel_type=args.kernel_type,
            )

    if args.load_model:
        model = model_init(False)
        print('Reloading model from ', args.log_dir)
        model.load_state_dict(torch.load(
            os.path.join(args.log_dir, 'model.pth')))
        return model

    stimulus = chirp.astype(np.float32)
    responses = [y_local.astype(np.float32), y_global.astype(np.float32)]
    if args.random_search:
        while True:
            t0 = time.time()
            config = get_sample()
            if config['hash'] in os.listdir(args.log_dir):
                # search for unused hash
                while True:
                    config['hash'] += 1
                    if config['hash'] not in os.listdir(args.log_dir):
                        break
            log_dir = os.path.join(args.log_dir, str(config['hash']))
            os.mkdir(log_dir)
            with open(os.path.join(log_dir, "args"), "w") as f:
                json.dump(args.__dict__, f)
            with open(os.path.join(log_dir, "config"), "w") as f:
                json.dump(config, f)
            print(config['hash'])
            model = model_init(config['load_best_lnr'], seed=config['hash'],
                               noise_init_scale=config['noise_init_scale'])

            if config['load_best_previous']:
                model.load_state_dict(
                    torch.load('best_bcn/z_scored_inaff/model.pth'))

            success = train(
                model,
                stimulus,
                responses,
                log_dir=log_dir,
                log_interval=args.log_interval,
                time_reg_weight=config['time_reg_weight'],
                sparsity_reg_weight=config['sparsity_reg_weight'],
                scaling_mean_weight=config['scaling_mean_weight'],
                scaling_std_weight=config['scaling_std_weight'],
                lr=config['lr'],
                max_steps=config['max_steps'],
                decrease_lr_after=config['decrease_lr_after'],
                stop_after=config['stop_after'],
                noise_scale=config['noise_scale'],
            )
            if success:
                print('done in %.2fs' % (time.time() - t0))
            else:
                print('failed in %.2fs' % (time.time() - t0))
                shutil.rmtree(log_dir)
    else:
        train(
            model_init(args.load_best_lnr),
            stimulus,
            responses,
            log_dir=args.log_dir,
            log_file=log_file,
            log_interval=args.log_interval,
            max_steps=args.max_steps,
            time_reg_weight=args.time_reg_weight,
            sparsity_reg_weight=args.sparsity_reg_weight,
            lr=args.lr,
            decrease_lr_after=args.decrease_lr_after,
            stop_after=args.stop_after,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IPL Network Models')
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--gpu", type=str, default="",
                        help='logging for random search')
    parser.add_argument('--log_dir', required=True, type=str,
                        help='where to save experiments')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='how many steps to optimize')
    parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                        help='how many steps to wait before logging training '
                             'status')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='whether to load model instead of train')
    parser.add_argument('--random_search', action='store_true', default=False,
                        help='whether to random search params instead of train')
    parser.add_argument('--load_best_lnr', action='store_true', default=False,
                        help='whether to load lnr model best parameters')
    parser.add_argument('--time_reg_weight', type=float, default=1e0,
                        help='weight for biphasic kernel regularization')
    parser.add_argument('--sparsity_reg_weight', type=float, default=1e-4,
                        help='sparsity weight for feedback weights')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='initial learning rate')
    parser.add_argument('--decrease_lr_after', type=int, default=3,
                        help='how many steps to worsen before decrease lr')
    parser.add_argument('--stop_after', type=int, default=5,
                        help='how many steps to lower lr before stop')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--noise_init_scale', type=float, default=1e-2,
                        help='initial noise variance')
    parser.add_argument('--no_random_init', action='store_true', default=False,
                        help='whether not to initialize parameters randomly')
    parser.add_argument('--fix_release_parameters', action='store_true',
                        default=False,
                        help='whether to fix lnr model ... parameters')
    parser.add_argument('--fix_linear_filter', action='store_true',
                        default=False,
                        help='whether to fix lnr model ... parameters')
    parser.add_argument('--fix_non_linearity', action='store_true',
                        default=False,
                        help='whether to fix lnr model ... parameters')
    parser.add_argument('--fix_steady_state', action='store_true',
                        default=False,
                        help='whether to fix lnr model ... parameters')
    parser.add_argument('--kernel_type', type=str, default='biphasic',
                        help='what kind of kernel to use in BCN (nonparametric')
    parser.add_argument('--chirp', type=str, default='new_jonathan',
                        help='which stimulus to use.')

    args = parser.parse_args()

    if args.random_search:
        try:
            os.mkdir(args.log_dir)
            with open(os.path.join(args.log_dir, "args"), "w") as f:
                json.dump(args.__dict__, f)
        except Exception:
            pass
        log_file = os.path.join(
                args.log_dir, "random_search_log_%s.txt" % args.gpu)
        with open( log_file, 'w') as f:
            print('Starting Training\n', file=f)
        log_file = open(log_file, 'a', 1)
        sys.stdout = log_file
        try:
            main(args)
        except Exception:
            traceback.print_exc(file=log_file)
        finally:
            log_file.close()

    else:  # single training
        # remove existing log_dir and save params
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.mkdir(args.log_dir)
        with open(os.path.join(args.log_dir, "args"), "w") as f:
            json.dump(args.__dict__, f)
        log_file = os.path.join(args.log_dir, "train_log.txt")
        with open(log_file, 'w') as f:
            print('Starting Training\n', file=f)
        log_file = open(log_file, 'a', 1)
        sys.stdout = log_file
        try:
            main(args, log_file)
        except Exception:
            traceback.print_exc(file=log_file)
        finally:
            log_file.close()

'''
run script e.g. like:
python3 training.py --max_steps=10

this will produce:
model.pth - saved model file
model_prediction.png - plot of model prediction vs true response
train_log.txt - plain txt file that tracks optimization
'''
