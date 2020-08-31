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

parser.add_argument('--load_path', type=str, default='pretrained/2spiarls/checkpt.pth')

# for the proposed method
parser.add_argument('--std_min', type=float, default=0.0)
parser.add_argument('--std_max', type=float, default=0.1)
parser.add_argument('--std_weight', type=float, default=2)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# loggera
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


def save_fig(samples, color, marker, dot_size, save_path, file_name):
    LOW = -3.5
    HIGH = 3.5
    fig = plt.figure(figsize=(3, 3))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(samples[:, 0], samples[:, 1], s=dot_size, color=color, marker=marker)
    ax.set_xlim(LOW, HIGH)
    ax.set_ylim(LOW, HIGH)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['bottom'].set_alpha(0.0)
    ax.spines['top'].set_alpha(0.0)
    ax.spines['right'].set_alpha(0.0)
    ax.spines['left'].set_alpha(0.0)

    COLOR_BACK = 0/255, 32/255, 64/255, 1.0
    ax.set_facecolor(COLOR_BACK)
    fig_filename = os.path.join(save_path, file_name+'.png')
    utils.makedirs(os.path.dirname(fig_filename))
    plt.savefig(fig_filename, format='png', dpi=1200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    save_path = 'generate1/' + args.data
    n_samples = 2000
    COLOR = 1, 1, 1, 0.64

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sample_real = toy_data.inf_train_gen(args.data, batch_size=n_samples)
    save_fig(sample_real, COLOR, 'D', 0.1, save_path, 'sample_data')

    softflow = build_model_tabular(args, 2).to(device)
    softflow_path = args.load_path
    ckpt_softflow = torch.load(softflow_path)
    softflow.load_state_dict(ckpt_softflow['state_dict'])
    softflow.eval()

    z = torch.randn(n_samples, 2).type(torch.float32).to(device)
    sample_s = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    with torch.no_grad():
        for ii in torch.split(inds, int(100**2)):
            zeros_std = torch.zeros(z[ii].shape[0], 1).to(z)
            sample_s.append(softflow(z[ii], zeros_std, reverse=True))
    sample_s = torch.cat(sample_s, 0).cpu().numpy()

    save_fig(sample_s, COLOR, 'D', 0.1, save_path, 'sample_softflow')

    print('done!')



