#! /bin/bash

cates="airplane"
for (( i = 1000; i <= 15000; i=i+1000)); do
    python train.py \
        --distributed \
        --cates ${cates} \
        --epoch ${i} \
        --batch_size 128 \
        --n_flow 12 \
        --multi_freq 4 \
        --n_flow_AF 9 \
        --h_dims_AF 256-256-256 \
        --save_freq 400 \
        --valid_freq 100 \
        --viz_freq 400 \
        --log_freq 1
    echo "Done"
done