import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import diffeq_layers
from .squeeze import squeeze, unsqueeze

__all__ = ["ODEnet", "ODEfunc"]


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self, hidden_dims, input_shape, strides, conv, layer_type="concatsquash", nonlinearity="softplus", num_squeeze=0
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "concatsquash": diffeq_layers.ConcatSquashConv2d
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "concatsquash": diffeq_layers.ConcatSquashLinear
            }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, tc, y):
        dx = y
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(tc, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx


class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 3
        y = states[0]
        std = states[1]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            std.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[3:]:
                s_.requires_grad_(True)
                
            tc = torch.cat([t, std.view(y.shape[0], -1)], dim=1)
            dy = self.diffeq(tc, y, *states[3:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        return tuple([dy, torch.zeros_like(std).requires_grad_(True), -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[3:]])
