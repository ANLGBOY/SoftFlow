import argparse
import os

def add_args(parser):

    # Prior Flow(glow) architecture
    parser.add_argument('--n_group', type=int, default=8)
    parser.add_argument('--n_flow', type=int, default=12)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_channels', type=int, default=256)
    parser.add_argument('--multi_freq', type=int, default=4)
    parser.add_argument('--multi_out', type=int, default=2)

    # Decoder Flow(Autoregressive Flow) architecture
    parser.add_argument('--h_dims_AF', type=str, default='128-128-128')
    parser.add_argument('--n_flow_AF', type=int, default=12)

    # input noise
    parser.add_argument('--std_min', type=float, default=0.0)
    parser.add_argument('--std_max', type=float, default=0.075)
    parser.add_argument('--test_std_n', type=float, default=0.0)
    parser.add_argument('--test_std_z', type=float, default=1.0)
    parser.add_argument('--std_scale', type=float, default=2)

    # training configuration
    parser.add_argument('--use_deterministic_encoder', action='store_true',
                        help='Whether to use a deterministic encoder.')
    parser.add_argument('--zdim', type=int, default=128,
                        help='Dimension of the shape code')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (of datasets) for training')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--step_size', type=int, default=5000,
                        help='Step size for scheduler.')
    parser.add_argument('--gamma', type=int, default=0.25,
                        help='Learning rate decay ratio.')
    parser.add_argument('--stop_scheduler', type=int, default=15000,
                        help='When to freeze leraning rate.')
    parser.add_argument('--epochs', type=int, default=15000,
                        help='Number of epochs for training (default: 12000)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to the checkpoint to be loaded for training.')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                    help='Path to the checkpoint to be loaded for genration and test.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Seed for initializing training. ')

    # data options
    parser.add_argument('--cates', type=str, nargs='+', default=["airplane"],
                        help="Categories to be trained")
    parser.add_argument('--data_dir', type=str, default="data/ShapeNetCore.v2.PC15k",
                        help="Path to the training data")
    parser.add_argument('--dataset_scale', type=float, default=1.,
                        help='Scale of the dataset (x,y,z * scale = real output, default=1).')
    parser.add_argument('--random_rotate', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--normalize_per_shape', action='store_true',
                        help='Whether to perform normalization per shape.')
    parser.add_argument('--normalize_std_per_axis', action='store_true',
                        help='Whether to perform normalization per axis.')
    parser.add_argument("--tr_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (train)')
    parser.add_argument("--te_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (test)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading threads')

    # logging and saving frequency
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--viz_freq', type=int, default=400)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:5555', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # Evaluation options
    parser.add_argument('--evaluate_recon', default=False, action='store_true',
                        help='Whether set to the evaluation for reconstruction.')
    parser.add_argument('--num_sample_shapes', default=10, type=int,
                        help='Number of shapes to be sampled (for demo.py).')
    parser.add_argument('--num_sample_points', default=2048, type=int,
                        help='Number of points (per-shape) to be sampled (for demo.py).')

    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='SoftPointFlow Experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join("results", args.cates[0], "SoftPointFlow")

    return args
