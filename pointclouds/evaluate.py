from datasets import get_testset
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import SoftPointFlow
import os
import torch
import numpy as np
import torch.nn as nn
import json

def get_test_loader(args):
    te_dataset = get_testset(args)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def evaluate_gen(itr, model, results_mva, log, log_mva, args):
    print('---- %dth evaluation ----' % itr)
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []

    std_in = args.test_std_n / args.std_max * args.std_scale
 
    for data in loader:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc = model.sample(B, N, std_in=std_in, std_z=args.test_std_z)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)
    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    if itr == 1:
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in metrics.items()}
        results['JSD'] = jsd
        results['itr'] = itr
        results_mva = results
    else:
        results = {}
        for k, v in metrics.items():
            if not isinstance(v, float):
                v = v.cpu().detach().item()
            results[k] = v
            results_mva[k] = (results_mva[k] * (itr-1) + v) / itr

        results['JSD'] = jsd
        results_mva['JSD'] = (results_mva['JSD'] * (itr-1) + jsd) / itr
        results['itr'] = itr
        results_mva['itr'] = itr

    log.write(json.dumps(results) + '\n')
    log_mva.write(json.dumps(results_mva) + '\n')
    log.flush()
    log_mva.flush()

    pprint(results_mva)

    return results_mva


def main(args):
    model = SoftPointFlow(args)
    
    log_path = 'test/' + args.cates[0]
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = open(os.path.join(log_path, 'SoftFlow_test.txt'), 'a')
    log_mva = open(os.path.join(log_path, 'SoftFlow_test_mva.txt'), 'a')
    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)
    print("Load Path:%s" % args.load_checkpoint)
    print('test_std_n:', args.test_std_n)
    print('test_std_z:', args.test_std_z)
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['model'])
    model.set_initialized(True)
    model.eval()

    with torch.no_grad():
        # Evaluate generation
        print('evaluate gen')
        results_mva = {}
        for i in range(1, 16 + 1):
            results_mva = evaluate_gen(i, model, results_mva, log, log_mva, args)
    print('Done!')

if __name__ == '__main__':
    args = get_args()
    main(args)
