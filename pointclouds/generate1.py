from datasets import get_trainset, get_testset, init_np_seed
from args import get_args
from pprint import pprint
from collections import defaultdict
from models.networks import SoftPointFlow

import os
import torch
import numpy as np
import torch.nn as nn
from utils import visualize_point_clouds
import imageio
import matplotlib.pyplot as plt


NUM_SAMPLE = 8
NUM_ITR = 2

def viz_save_recon(gtr_batch,  pts_SPF_batch, file_path, color, resize_scale, pert_order=[0, 1, 2]):
    B, _, _ = gtr_batch.shape
    fig = plt.figure(figsize=(3*NUM_SAMPLE, 6))

    for i in range(B):
        gtr, pts_SPF = gtr_batch[i], pts_SPF_batch[i]
        gtr = gtr.cpu().detach().numpy()[:, pert_order]
        pts_SPF = pts_SPF.cpu().detach().numpy()[:, pert_order]

        ax1 = fig.add_subplot(2,NUM_SAMPLE,i+1, projection='3d')
        ax1.set_title("Ground Truth")
        ax1.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], c=color, s=0.2, marker='D')
        
        ax2 = fig.add_subplot(2,NUM_SAMPLE,NUM_SAMPLE+i+1, projection='3d')
        ax2.set_title("SoftPointFlow")
        ax2.scatter(pts_SPF[:, 0], pts_SPF[:, 1], pts_SPF[:, 2], c=color, s=0.2, marker='D')
        
        resize_xlim = (ax1.get_xlim()[0] * resize_scale, ax1.get_xlim()[1] * resize_scale) 
        resize_ylim = (ax1.get_ylim()[0] * resize_scale, ax1.get_ylim()[1] * resize_scale) 
        resize_zlim = (ax1.get_zlim()[0] * resize_scale, ax1.get_zlim()[1] * resize_scale) 

        ax1.set_xlim(resize_xlim)
        ax1.set_ylim(resize_ylim)
        ax1.set_zlim(resize_zlim)

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_zlim(ax1.get_zlim())

        ax1.axis('off')
        ax2.axis('off')

    fig.canvas.draw()

    plt.savefig(file_path, format='png', dpi=300,  bbox_inches='tight')

    plt.close()


def viz_save_sample(pts_SPF_batch, file_path, color, resize_scale, pert_order=[0, 1, 2]):
    B, _, _ = pts_SPF_batch.shape
    fig = plt.figure(figsize=(3*NUM_SAMPLE, 3))

    for i in range(B):
        pts_SPF = pts_SPF_batch[i]
        pts_SPF = pts_SPF.cpu().detach().numpy()[:, pert_order]

        ax = fig.add_subplot(1,NUM_SAMPLE,i+1, projection='3d')
        ax.scatter(pts_SPF[:, 0], pts_SPF[:, 1], pts_SPF[:, 2], c=color, s=0.2, marker='D')

        resize_xlim = (ax.get_xlim()[0] * resize_scale, ax.get_xlim()[1] * resize_scale) 
        resize_ylim = (ax.get_ylim()[0] * resize_scale, ax.get_ylim()[1] * resize_scale) 
        resize_zlim = (ax.get_zlim()[0] * resize_scale, ax.get_zlim()[1] * resize_scale) 
        ax.set_xlim(resize_xlim)
        ax.set_ylim(resize_ylim)
        ax.set_zlim(resize_zlim)

        ax.axis('off')

    fig.canvas.draw()

    plt.savefig(file_path, format='png', dpi=300,  bbox_inches='tight')

    plt.close()
    

def gen_samples(softpointflow, save_dir, color, resize_scale, args):
    tr_dataset = get_trainset(args)
    te_dataset = get_testset(args)

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=NUM_SAMPLE, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=NUM_SAMPLE, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    std_in = args.test_std_n / args.std_max * args.std_scale
    for bidx, data in enumerate(train_loader):
        idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
        seen_input = tr_batch.cuda(args.gpu, non_blocking=True)
        samples_SPF = softpointflow.reconstruct(seen_input, std_in=std_in, std_z=args.test_std_z)
        file_path = os.path.join(save_dir, 'seen', 'result_seen_{}.png'.format(bidx+1))
        result = viz_save_recon(seen_input, samples_SPF, file_path, color, resize_scale, pert_order=train_loader.dataset.display_axis_order)

        print('%dth image of seen data is saved!' %(bidx+1))
        if bidx == NUM_ITR-1:
            break

    for bidx, data in enumerate(test_loader):
        idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
        test_input = tr_batch.cuda(args.gpu, non_blocking=True)
        samples_SPF = softpointflow.reconstruct(test_input, std_in=std_in, std_z=args.test_std_z)
        file_path = os.path.join(save_dir, 'unseen', 'result_unseen_{}.png'.format(bidx+1))
        result = viz_save_recon(test_input, samples_SPF, file_path, color, resize_scale, pert_order=train_loader.dataset.display_axis_order)

        print('%dth image of unseen data is saved!' %(bidx+1))
        if bidx == NUM_ITR-1:
            break

    for bidx in range(NUM_ITR):
        num_points = args.tr_max_sample_points
        _, samples_SPF = softpointflow.sample(NUM_SAMPLE, num_points, std_in=std_in, std_z=args.test_std_z)
        file_path = os.path.join(save_dir, 'sample', 'result_sample_{}.png'.format(bidx+1))
        result = viz_save_sample(samples_SPF, file_path, color, resize_scale, pert_order=train_loader.dataset.display_axis_order)
        print('%dth image of sampling is saved!' %(bidx+1))


def get_viz_config(cates):
    if cates == 'airplane':
        R = 180/255
        G = 30/255
        B = 45/255
        alpha = 0.4
        resize_scale = 0.9
    
    elif cates == 'car':
        R = 35/255
        G = 110/255
        B = 20/255
        alpha = 0.4
        resize_scale = 0.9

    elif cates == 'chair':
        R = 30/255
        G = 45/255
        B = 150/255
        alpha = 0.4
        resize_scale = 0.9

    color = [[R,G,B,alpha]]

    return color, resize_scale

def main(args):
    cates = ""
    for cate in args.cates:
        cates += cate

    save_dir = os.path.join("generate1/"+cates)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'/seen')
        os.makedirs(save_dir+'/unseen')
        os.makedirs(save_dir+'/sample')

    color, resize_scale = get_viz_config(cates)

    softpointflow = SoftPointFlow(args)
    def _transform_(m):
        return nn.DataParallel(m)

    softpointflow = softpointflow.cuda()
    softpointflow.multi_gpu_wrapper(_transform_)

    print("load_checkpoint:%s" % args.load_checkpoint)
    checkpoint = torch.load(args.load_checkpoint)
    softpointflow.load_state_dict(checkpoint["model"])
    softpointflow.set_initialized(True)
    softpointflow.eval()

    with torch.no_grad():
        gen_samples(softpointflow, save_dir, color, resize_scale, args)


if __name__ == '__main__':
    args = get_args()
    main(args)
