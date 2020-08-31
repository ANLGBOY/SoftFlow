import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.backends import cudnn
from torch import optim
from tensorboardX import SummaryWriter
import sys
import os
import warnings
import numpy as np
import imageio
import random
import faulthandler
import time
import gc
from models.networks import SoftPointFlow
from args import get_args
from utils import AverageValueMeter, set_random_seed, apply_random_rotation, save, resume, visualize_point_clouds
from datasets import get_trainset, get_testset, init_np_seed

faulthandler.enable()

def main_worker(gpu, save_dir, ngpus_per_node, init_data, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # resume training!!!
    #################################
    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
        print('Checkpoint is set to the latest one.')
    #################################

    # multi-GPU setup
    model = SoftPointFlow(args)
    if args.distributed:  # Multiple processes, single GPU per process
        if args.gpu is not None:
            def _transform_(m):
                return nn.parallel.DistributedDataParallel(
                    m, device_ids=[args.gpu], output_device=args.gpu, check_reduction=True)

            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.multi_gpu_wrapper(_transform_)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = 0
        else:
            assert 0, "DistributedDataParallel constructor should always set the single device scope"
    else:  # Single process, multiple GPUs per process
        def _transform_(m):
            return nn.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    start_epoch = 1
    valid_loss_best = 987654321
    optimizer = model.make_optimizer(args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.resume_checkpoint is not None:
        model, optimizer, scheduler, start_epoch, valid_loss_best, log_dir = resume(
            args.resume_checkpoint, model, optimizer, scheduler)
        model.set_initialized(True)
        print('Resumed from: ' + args.resume_checkpoint)

    else:
        log_dir = save_dir + "/runs/" + str(time.strftime('%Y-%m-%d_%H:%M:%S'))
        with torch.no_grad():
            inputs, inputs_noisy, std_in = init_data
            inputs = inputs.to(args.gpu, non_blocking=True)
            inputs_noisy = inputs_noisy.to(args.gpu, non_blocking=True)
            std_in = std_in.to(args.gpu, non_blocking=True)
            _ = model(inputs, inputs_noisy, std_in, optimizer,  None, None, init=True)
        del inputs, inputs_noisy, std_in
        print('Actnorm is initialized')

    if not args.distributed or (args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(logdir=log_dir)
    else:
        writer = None

    # initialize datasets and loaders
    tr_dataset = get_trainset(args)
    te_dataset = get_testset(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(te_dataset)
    else:
        train_sampler = None
        test_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_np_seed)

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=0, pin_memory=True, sampler=test_sampler, drop_last=True,
        worker_init_fn=init_np_seed)

    # save dataset statistics
    if not args.distributed or (args.rank % ngpus_per_node == 0):
        np.save(os.path.join(save_dir, "train_set_mean.npy"), tr_dataset.all_points_mean)
        np.save(os.path.join(save_dir, "train_set_std.npy"), tr_dataset.all_points_std)
        np.save(os.path.join(save_dir, "train_set_idx.npy"), np.array(tr_dataset.shuffle_idx))
    
    # main training loop
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    seen_inputs = next(iter(train_loader))['train_points'].cuda(args.gpu, non_blocking=True)
    unseen_inputs = next(iter(test_loader))['test_points'].cuda(args.gpu, non_blocking=True)
    del test_loader

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs+1):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if writer is not None:
            writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        model.train()
        # train for one epoch
        
        for bidx, data in enumerate(train_loader):
            step = bidx + len(train_loader) * (epoch - 1)
            tr_batch = data['train_points']
            if args.random_rotate:
                tr_batch, _, _ = apply_random_rotation(
                    tr_batch, rot_axis=train_loader.dataset.gravity_axis)

            inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            B, N, D = inputs.shape
            std = (args.std_max - args.std_min) * torch.rand_like(inputs[:,:,0]).view(B,N,1) + args.std_min

            eps = torch.randn_like(inputs) * std
            std_in = std / args.std_max * args.std_scale
            inputs_noisy = inputs + eps
            out = model(inputs, inputs_noisy, std_in, optimizer, step, writer)
            entropy, prior_nats, recon_nats, loss = out['entropy'], out['prior_nats'], out['recon_nats'], out['loss']
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                if writer is not None:
                    writer.add_scalar('train/avg_time', duration, step)
                print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Entropy %2.5f LatentNats %2.5f PointNats %2.5f loss %2.5f"
                      % (args.rank, epoch, bidx, len(train_loader), duration, entropy,
                         prior_nats, recon_nats, loss))
            del inputs, inputs_noisy, std_in, out, eps
            gc.collect()

        if epoch < args.stop_scheduler:
            scheduler.step()

        if epoch % args.valid_freq == 0:
            with torch.no_grad():
                model.eval()
                valid_loss = 0.0
                valid_entropy = 0.0
                valid_prior = 0.0
                valid_prior_nats = 0.0
                valid_recon = 0.0
                valid_recon_nats = 0.0
                for bidx, data in enumerate(train_loader):
                    step = bidx + len(train_loader) * epoch
                    tr_batch = data['test_points']
                    if args.random_rotate:
                        tr_batch, _, _ = apply_random_rotation(
                            tr_batch, rot_axis=train_loader.dataset.gravity_axis)

                    inputs = tr_batch.cuda(args.gpu, non_blocking=True)
                    B, N, D = inputs.shape
                    std = (args.std_max - args.std_min) * torch.rand_like(inputs[:,:,0]).view(B,N,1) + args.std_min

                    eps = torch.randn_like(inputs) * std
                    std_in = std / args.std_max * args.std_scale
                    inputs_noisy = inputs + eps
                    out = model(inputs, inputs_noisy, std_in, optimizer, step, writer, valid=True)
                    valid_loss += out['loss'] / len(train_loader)
                    valid_entropy += out['entropy'] / len(train_loader)
                    valid_prior += out['prior'] / len(train_loader)
                    valid_prior_nats += out['prior_nats'] / len(train_loader)
                    valid_recon += out['recon'] / len(train_loader)
                    valid_recon_nats += out['recon_nats'] / len(train_loader)
                    del inputs, inputs_noisy, std_in, out, eps
                    gc.collect()

                if writer is not None:
                    writer.add_scalar('valid/entropy', valid_entropy, epoch)
                    writer.add_scalar('valid/prior', valid_prior, epoch)
                    writer.add_scalar('valid/prior(nats)', valid_prior_nats, epoch)
                    writer.add_scalar('valid/recon', valid_recon, epoch)
                    writer.add_scalar('valid/recon(nats)', valid_recon_nats, epoch)
                    writer.add_scalar('valid/loss', valid_loss, epoch)
                
                duration = time.time() - start_time
                start_time = time.time()
                print("[Valid] Epoch %d Time [%3.2fs] Entropy %2.5f LatentNats %2.5f PointNats %2.5f loss %2.5f loss_best %2.5f"
                    % (epoch, duration, valid_entropy, valid_prior_nats, valid_recon_nats, valid_loss, valid_loss_best))
                if valid_loss < valid_loss_best:
                    valid_loss_best = valid_loss
                    if not args.distributed or (args.rank % ngpus_per_node == 0):
                        save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir,
                            os.path.join(save_dir, 'checkpoint-best.pt'))
                        print('best model saved!')

        if epoch % args.save_freq == 0 and (not args.distributed or (args.rank % ngpus_per_node == 0)):
            save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir,
                os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir,
                os.path.join(save_dir, 'checkpoint-latest.pt'))
            print('model saved!')

        # save visualizations
        if epoch % args.viz_freq == 0:
            with torch.no_grad():
                # reconstructions
                model.eval()
                samples = model.reconstruct(unseen_inputs)
                results = []
                for idx in range(min(16, unseen_inputs.size(0))):
                    res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
                                                pert_order=train_loader.dataset.display_axis_order)

                    results.append(res)
                res = np.concatenate(results, axis=1)
                imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_recon_unseen.png' % (epoch, args.gpu)),
                                res.transpose(1, 2, 0))
                if writer is not None:
                    writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)

                samples = model.reconstruct(seen_inputs)
                results = []
                for idx in range(min(16, seen_inputs.size(0))):
                    res = visualize_point_clouds(samples[idx], seen_inputs[idx], idx,
                                                pert_order=train_loader.dataset.display_axis_order)

                    results.append(res)
                res = np.concatenate(results, axis=1)
                imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_recon_seen.png' % (epoch, args.gpu)),
                                res.transpose(1, 2, 0))
                if writer is not None:
                    writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)

                num_samples = min(16, unseen_inputs.size(0))
                num_points = unseen_inputs.size(1)
                _, samples = model.sample(num_samples, num_points)
                results = []
                for idx in range(num_samples):
                    res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
                                                pert_order=train_loader.dataset.display_axis_order)
                    results.append(res)
                res = np.concatenate(results, axis=1)
                imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_sample.png' % (epoch, args.gpu)),
                                res.transpose((1, 2, 0)))
                if writer is not None:
                    writer.add_image('tr_vis/sampled', torch.as_tensor(res), epoch)
                
                print('image saved!')


def get_init_data(args):
    tr_dataset = get_trainset(args)
    init_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=128, shuffle=None, 
        pin_memory=True, sampler=None, drop_last=True, worker_init_fn=init_np_seed)
        
    data = next(iter(init_loader))
    inputs = data['train_points']
    B, N, D = inputs.shape
    std = (args.std_max - args.std_min) * torch.rand_like(inputs[:,:,0]).view(B,N,1) + args.std_min
    eps = torch.randn_like(inputs) * std
    std_in = std / args.std_max * args.std_scale
    inputs_noisy = inputs + eps

    return (inputs, inputs_noisy, std_in)


def main():
    args = get_args()
    set_random_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'images'))

    with open(os.path.join(args.save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    init_data = get_init_data(args)
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args.save_dir, ngpus_per_node, init_data, args))
    else:
        main_worker(args.gpu, args.save_dir, ngpus_per_node, init_data, args)


if __name__ == '__main__':
    main()
