#! /bin/bash


cates="chair"
test_std_n=0.00

python generate2.py \
    --cates ${cates} \
    --load_checkpoint pretrained/${cates}/checkpoint-best.pt \
    --te_max_sample_points 10000 \
    --tr_max_sample_points 10000 \
    --n_flow 12 \
    --multi_freq 4 \
    --n_flow_AF 9 \
    --h_dims_AF 256-256-256 \
    --test_std_n ${test_std_n}
