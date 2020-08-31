import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import pi, log
import torch.nn as nn

logabs = lambda x: torch.log(torch.abs(x))

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
        x = x.permute(0,2,1)
        if reverse:
            return self.reverse(x)
        B, _, T = x.size()

        if not self.is_initialized:
            self.initialize(x)
            self.is_initialized = True

        log_abs = logabs(self.scale)

        logdet = torch.ones_like(x[:,0,:]) * torch.sum(log_abs)
        x = self.scale * (x + self.loc)
        x = x.permute(0,2,1)
        return x, logdet

    def reverse(self, output):
        output = output / self.scale - self.loc
        output = output.permute(0,2,1)
        return output


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
        z = z.permute(0,2,1)
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()
        if reverse:
            # if not hasattr(self, 'W_inverse'):
                # Reverse computation
            W_inverse = W.float().inverse()
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == 'torch.cuda.HalfTensor':
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            z = z.permute(0,2,1)
            return z
        else:
            # Forward computation
            log_det_W = torch.ones_like(z[:,0,:]) * torch.logdet(W)
            z = self.conv(z)
            z = z.permute(0,2,1)

            return z, log_det_W


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

        self._hyper_bias = nn.Linear(dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_c, dim_out)

        if dim_out ==2:
            self._layer.weight.data.zero_()
            self._layer.bias.data.zero_()

            self._hyper_bias.weight.data.zero_()

            self._hyper_gate.weight.data.zero_()
            self._hyper_gate.bias.data.zero_()

    def forward(self, context, x):
        input_gate = self._hyper_gate(context).unsqueeze(1)
        bias = self._hyper_bias(context).unsqueeze(1)
        gate = torch.sigmoid(input_gate)

        ret = self._layer(x) * gate + bias

        return ret


class Flow(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, context_dim, h_dims):
        super(Flow, self).__init__()

        # build models and add them
        layers1 = []
        activation_fns = []
        hidden_shape = (2,)
        for dim_out in (h_dims+ (2,)):
            layer = ConcatSquashLinear(hidden_shape[0], dim_out, context_dim)
            layers1.append(layer)
            activation_fns.append(nn.Tanh())

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out

        self.layers1 = nn.ModuleList(layers1)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

        layers2 = []
        activation_fns = []
        hidden_shape = (3,)
        for dim_out in (h_dims +(2,)):
            layer = ConcatSquashLinear(hidden_shape[0], dim_out, context_dim)
            layers2.append(layer)
            activation_fns.append(nn.Tanh())

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out

        self.layers2 = nn.ModuleList(layers2)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, x_in, std_in, z, reverse=False):
        if reverse:
            x_a = torch.cat((x_in[:,:,0:1], std_in), dim=2)
            for l, layer in enumerate(self.layers1):
                x_a = layer(z, x_a)
                # if not last layer, use nonlinearity
                if l < len(self.layers1) - 1:
                    x_a = self.activation_fns[l](x_a)
            log_s = x_a[:,:,0:1]
            b = x_a[:,:,1:2]
            x_in[:,:,1:2]  = torch.exp(-log_s)*(x_in[:,:,1:2] - b)

            x_b = torch.cat((x_in[:,:,0:2], std_in), dim=2)
            for l, layer in enumerate(self.layers2):
                x_b = layer(z, x_b)
                # if not last layer, use nonlinearity
                if l < len(self.layers2) - 1:
                    x_b = self.activation_fns[l](x_b)
            log_s = x_b[:,:,0:1]
            b = x_b[:,:,1:2]
            x_in[:,:,2:3]  = torch.exp(-log_s)*(x_in[:,:,2:3] - b)

            return x_in

        else:
            x_a = torch.cat((x_in[:,:,0:1], std_in), dim=2)
            x_b = torch.cat((x_in[:,:,0:2], std_in), dim=2)

            for l, layer in enumerate(self.layers1):
                x_a = layer(z, x_a)
                # if not last layer, use nonlinearity
                if l < len(self.layers1) - 1:
                    x_a = self.activation_fns[l](x_a)

            for l, layer in enumerate(self.layers2):
                x_b = layer(z, x_b)
                # if not last layer, use nonlinearity
                if l < len(self.layers2) - 1:
                    x_b = self.activation_fns[l](x_b)

            log_s = torch.cat((x_a[:,:,0:1], x_b[:,:,0:1]), dim=2)
            b = torch.cat((x_a[:,:,1:2], x_b[:,:,1:2]), dim=2)

            return log_s, b


class AF(torch.nn.Module):
    def __init__(self, n_flows=8, z_dim=128, h_dims=(256, 256, 256)):
        super(AF, self).__init__()
        self.n_flows = n_flows
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.actnorm = torch.nn.ModuleList()

        for k in range(n_flows):
            self.actnorm.append(ActNorm(3))
            self.convinv.append(Invertible1x1Conv(3))
            self.WN.append(Flow(z_dim, h_dims))

    def set_initialized(self, switch):
        for actnorm in self.actnorm:
            actnorm.is_initialized=switch

    def forward(self, x, std_in, z, reverse=False):
        if reverse:
            return self.reverse(x, std_in, z)

        log_s_list = []
        log_det_W_list = []
        log_det_a_list = []
        output_z = []
        for k in range(self.n_flows):
            x, log_det_a = self.actnorm[k](x)
            x, log_det_W = self.convinv[k](x)
            log_det_W_list.append(log_det_W)
            log_det_a_list.append(log_det_a)

            x_in = x[:,:,:-1]
            log_s, b = self.WN[k](x_in, std_in, z)
            x_out= torch.exp(log_s)*x[:,:,1:] + b
            x=torch.cat((x[:,:,0:1], x_out), dim=2)

            log_s_list.append(log_s)

        for i, log_s in enumerate(log_s_list):
            if i == 0:
                # print(log_s.shape)
                log_s_total = log_s.sum(dim=(2))
                log_det_W_total = log_det_W_list[i]
                log_det_a_total = log_det_a_list[i]
            else:
                log_s_total = log_s_total + log_s.sum(dim=(2))
                log_det_W_total = log_det_W_total + log_det_W_list[i]
                log_det_a_total = log_det_a_total + log_det_a_list[i]

        log_det = (log_s_total + log_det_W_total + log_det_a_total)

        return x, -log_det


    def reverse(self, z, std_in, c):
        for k in reversed(range(self.n_flows)):
            z = self.WN[k](z, std_in, c, reverse=True)
            z = self.convinv[k](z, reverse=True)
            z = self.actnorm[k](z, reverse=True)

        return z

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list