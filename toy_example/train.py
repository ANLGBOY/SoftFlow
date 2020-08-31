import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time

import torch
import torch.optim as optim

import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import standard_normal_logprob
from lib.utils import count_nfe, count_total_time
from lib.utils import build_model_tabular

from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc


SOLVERS = ["dopri5"]
parser = argparse.ArgumentParser('SoftFlow')
parser.add_argument(
    '--data', choices=['2spirals_1d','2spirals_2d', 'swissroll_1d','swissroll_2d', 'circles_1d', 'circles_2d', '2sines_1d', 'target_1d'],
    type=str, default='2spirals_1d'
)
parser.add_argument("--layer_type", type=str, default="concatsquash", choices=["concatsquash"])
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--niters', type=int, default=36000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# for the proposed method
parser.add_argument('--std_min', type=float, default=0.0)
parser.add_argument('--std_max', type=float, default=0.1)
parser.add_argument('--std_weight', type=float, default=2)

parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=400)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# logger
save_path = 'results/' + args.data + '/SoftFlow'
utils.makedirs(save_path)
logger = utils.get_logger(logpath=os.path.join(save_path, 'logs'), filepath=os.path.abspath(__file__))

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def get_transforms(model):

    def sample_fn(z, logpz=None):
        zeors_std = torch.zeros(z.shape[0], 1).to(z)
        if logpz is not None:
            return model(z, zeors_std, logpz, reverse=True)
        else:
            return model(z, zeors_std, reverse=True)

    def density_fn(x, logpx=None):
        zeors_std = torch.zeros(x.shape[0], 1).to(x)
        if logpx is not None:
            return model(x, zeors_std, logpx, reverse=False)
        else:
            return model(x, zeors_std, reverse=False)

    return sample_fn, density_fn


def compute_loss(args, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    # load data
    x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    std = (args.std_max - args.std_min) * torch.rand_like(x[:,0]).view(-1,1) + args.std_min
    eps = torch.randn_like(x) * std
    std_in = std / args.std_max * args.std_weight
    z, delta_logp = model(x+eps, std_in, zero)

    # compute log q(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


if __name__ == '__main__':

    model = build_model_tabular(args, 2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')
    model.train()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        loss = compute_loss(args, model)
        loss_meter.update(loss.item())

        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)

        loss.backward()
        optimizer.step()

        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)

        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            )
        )

        logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
                test_nfe = count_nfe(model)
                log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, test_loss, test_nfe)
                logger.info(log_message)

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    utils.makedirs(save_path)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(save_path, 'checkpt.pth'))
                model.train()

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                model.eval()
                p_samples = toy_data.inf_train_gen(args.data, batch_size=2000)

                sample_fn, density_fn = get_transforms(model)

                plt.figure(figsize=(9, 3))
                visualize_transform(
                    p_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
                    samples=True, npts=200, device=device
                )
                fig_filename = os.path.join(save_path, 'figs', '{:04d}.jpg'.format(itr))
                utils.makedirs(os.path.dirname(fig_filename))
                plt.savefig(fig_filename, format='png', dpi=1200)
                plt.close()
                model.train()

        end = time.time()

    logger.info('Training has finished.')
