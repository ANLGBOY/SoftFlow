import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import pi, log
import torch.nn as nn


logabs = lambda x: torch.log(torch.abs(x))

# @torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, n_channels):
    in_act = input_a
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts


class ActNorm(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))

        self.is_initialized = False

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x, reverse=False):
        if reverse:
            return self.reverse(x)
        B, _, T = x.size()

        if not self.is_initialized:
            self.initialize(x)
            self.is_initialized = True

        log_abs = logabs(self.scale)

        logdet = torch.ones_like(x[:,0,0]) * torch.sum(log_abs) * T

        return self.scale * (x + self.loc), logdet

    def reverse(self, output):
        return output / self.scale - self.loc


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        if reverse:
            W_inverse = W.float().inverse()
            W_inverse = Variable(W_inverse[..., None])
            self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
        
            log_det_W = torch.ones_like(z[:,0,0]) * n_of_groups * torch.logdet(W)
            z = self.conv(z)

            return z, log_det_W


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, n_layers, n_channels, kernel_size=3):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 1
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x):
        x = self.start(x)
        output = torch.zeros_like(x)
        for i in range(self.n_layers):
            acts = fused_add_tanh_sigmoid_multiply(self.in_layers[i](x), self.n_channels)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = x + res_skip_acts[:,:self.n_channels,:]
                output = output + res_skip_acts[:,self.n_channels:,:]
            else:
                output = output + res_skip_acts

        return self.end(output)


class Glow(torch.nn.Module):
    def __init__(self, n_group=4, n_flows=15, n_layers=4, n_channels=128,  multi_freq=5, multi_out=2):
        super(Glow, self).__init__()

        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.multi_freq = multi_freq
        self.multi_out = multi_out
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.actnorm = torch.nn.ModuleList()
        self.sigma = 1.0

        n_half = int(n_group/2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.multi_freq == 0 and k > 0:
                n_half = n_half - int(self.multi_out/2)
                n_remaining_channels = n_remaining_channels - self.multi_out
            self.actnorm.append(ActNorm(n_remaining_channels))
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_layers, n_channels))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def set_initialized(self, switch):
        for actnorm in self.actnorm:
            actnorm.is_initialized=switch

    def forward(self, x, reverse=False):
        if reverse:
            return self.reverse(x)

        B, D = x.shape
        x = x.view(B, self.n_group, -1)
        log_s_list = []
        log_det_W_list = []
        log_det_a_list = []
        z_list = []
        for k in range(self.n_flows):
            if k % self.multi_freq == 0 and k > 0:
                z_list.append(x[:,:self.multi_out,:])
                x = x[:,self.multi_out:,:]

            x, log_det_a = self.actnorm[k](x)
            x, log_det_W = self.convinv[k](x)

            log_det_W_list.append(log_det_W)
            log_det_a_list.append(log_det_a)

            n_half = int(x.size(1)/2)
            x_0 = x[:,:n_half,:]
            x_1 = x[:,n_half:,:]

            output = self.WN[k](x_0)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            x_out = torch.exp(log_s) * x_1 + b
            log_s_list.append(log_s)

            x = torch.cat((x_0, x_out), 1)

        z_list.append(x)
        z = torch.cat(z_list, dim=1).view(B, D)

        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = log_s.sum(dim=(1,2))
                log_det_W_total = log_det_W_list[i]
                log_det_a_total = log_det_a_list[i]
            else:
                log_s_total = log_s_total + log_s.sum(dim=(1,2))
                log_det_W_total = log_det_W_total + log_det_W_list[i]
                log_det_a_total = log_det_a_total + log_det_a_list[i]

        log_det = (log_det_a_total + log_s_total + log_det_W_total).unsqueeze(1)

        return z, -log_det


    def reverse(self, z):
        B, D = z.shape

        z = z.view(B, self.n_group, -1)
        z_list = []
        for k in range(self.n_flows):
            if k % self.multi_freq == 0 and k > 0:
                z_list.append(z[:,:self.multi_out, :])
                z = z[:,self.multi_out:,:]

        for k in reversed(range(self.n_flows)):
            n_half = int(z.size(1)/2)
            z_0 = z[:,:n_half,:]
            z_1 = z[:,n_half:,:]

            output = self.WN[k](z_0)

            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            z_1 = (z_1 - b) * torch.exp(-log_s)
            z = torch.cat([z_0, z_1],1)

            z = self.convinv[k](z, reverse=True)
            z = self.actnorm[k](z, reverse=True)

            if k % self.multi_freq == 0 and k > 0:
                z = torch.cat((z_list.pop(), z), dim=1)
                
        z = z.view(B,D)

        return z


    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list