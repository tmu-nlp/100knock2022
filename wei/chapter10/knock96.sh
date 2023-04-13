#!/usr/bin/env bash
# --tensorboard-logdirを使って可視化できる
fairseq-train knock95_subwords_sp \
    --fp16 \
    --tensorboard-logdir log96 \
    --save-dir save96 \
    --max-epoch 5 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.2 --weight-decay 0.0001 \
    --update-freq 1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 96.log