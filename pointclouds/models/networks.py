import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from utils import truncated_normal, reduce_tensor, standard_normal_logprob
from models.glow import Glow
from models.AF import AF

class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)

        return m, v


# Model
class SoftPointFlow(nn.Module):
    def __init__(self, args):
        super(SoftPointFlow, self).__init__()
        h_dims_AF = tuple(map(int, args.h_dims_AF.split("-")))

        self.input_dim = 3
        self.zdim = args.zdim
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.distributed = args.distributed
        self.encoder = Encoder(
                zdim=args.zdim, input_dim=3,
                use_deterministic_encoder=args.use_deterministic_encoder)
        self.latent_glow = Glow(args.n_group, args.n_flow, args.n_layers, args.n_channels, args.multi_freq, args.multi_out)
        self.point_AF = AF(args.n_flow_AF, args.zdim, h_dims_AF)

    @staticmethod
    def sample_gaussian(size, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.point_AF = f(self.point_AF)
        self.latent_glow = f(self.latent_glow)

    def make_optimizer(self, args):
        def _get_opt_(params):
            optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                    weight_decay=args.weight_decay)
            return optimizer
        opt = _get_opt_(list(self.encoder.parameters()) + list(self.point_AF.parameters())
                        + list(list(self.latent_glow.parameters())))
        return opt

    def set_initialized(self, switch):
        self.latent_glow.module.set_initialized(switch)
        self.point_AF.module.set_initialized(switch)
        print('is_initialized in actnorm is set to ' + str(switch))

    def forward(self, x, x_noisy, std_in, opt, step=None, writer=None, init=False, valid=False):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)

        # Compute H[Q(z|X)]
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size).to(z)
        else:
            entropy = self.gaussian_entropy(z_sigma)

        # Compute the prior probability P(z)
        w, delta_log_pw = self.latent_glow(z)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_pw = delta_log_pw.view(batch_size, 1)
        log_pz = log_pw - delta_log_pw

        # Compute the reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        y, delta_log_py = self.point_AF(x_noisy, std_in, z_new)
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss
        entropy_loss = -entropy.mean()
        recon_loss = -log_px.mean()
        prior_loss = -log_pz.mean()
        loss = entropy_loss + prior_loss + recon_loss
        if not init and not valid:
            loss.backward()
            opt.step()

        # LOGGING (after the training)
        if self.distributed:
            loss = reduce_tensor(loss.mean())
            entropy_log = reduce_tensor(entropy.mean())
            recon = reduce_tensor(-log_px.mean())
            prior = reduce_tensor(-log_pz.mean())
        else:
            loss = loss.mean()
            entropy_log = entropy.mean()
            recon = -log_px.mean()
            prior = -log_pz.mean()

        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.zdim)

        if writer is not None and not valid:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)
            writer.add_scalar('train/loss', loss.item(), step)

        return {
            'entropy': entropy_log.cpu().detach().item()
            if not isinstance(entropy_log, float) else entropy_log,
            'prior_nats': prior_nats,
            'recon_nats': recon_nats,
            'prior': prior,
            'recon': recon,
            'loss': loss.item()
        }

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, std_in, std_z):
        # transform points from the prior to a point cloud, conditioned on a shape code
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim)) * std_z
        std_in = torch.ones_like(y[:,:,0]).unsqueeze(2) * std_in
        x = self.point_AF(y, std_in, z, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, std_in=0.0, std_z=1.0, gpu=None):
        # Generate the shape code from the prior
        w = self.sample_gaussian((batch_size, self.zdim), gpu=gpu) * std_z
        z = self.latent_glow(w, reverse=True)
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), gpu=gpu) * std_z
        std_in = torch.ones_like(y[:,:,0]).unsqueeze(2) * std_in
        x = self.point_AF(y, std_in, z, reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, std_in=0.0, std_z=1.0, num_points=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, std_in, std_z)
        return x
