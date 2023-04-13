#!/usr/bin/env bash
# train a Transformer translation mode
# epochs=3
# specify which GPU to use
CUDA_VISIBLE_DEVICE=0

fairseq-train data91 \
    --fp16 \
    --save-dir save91 \
    --max-epoch 3 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --update-freq 1 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 91.log
# epoch 001: 100% 1758/1759 [10:52<00:00,  2.65it/s, loss=7.646, nll_loss=6.351, ppl=81.61, wps=18200.4, ups=2.67, wpb=6827.2, bsz=233.9, num_updates=1700, lr=0.00085, gnorm=0.886, clip=22, loss_scale=8, train_wall=37, gb_free=5.8, wall=633]
# epoch 002: 100% 1758/1759 [10:53<00:00,  2.71it/s, loss=6.818, nll_loss=5.399, ppl=42.21, wps=18216, ups=2.69, wpb=6772.2, bsz=242.2, num_updates=3500, lr=0.000755929, gnorm=0.583, clip=1, loss_scale=8, train_wall=37, gb_free=5.6, wall=1324]
# epoch 003: 100% 1758/1759 [10:52<00:00,  2.76it/s, loss=6.398, nll_loss=4.913, ppl=30.12, wps=18330.8, ups=2.7, wpb=6782.8, bsz=279.4, num_updates=5200, lr=0.000620174, gnorm=0.624, clip=4, loss_scale=8, train_wall=37, gb_free=7.3, wall=1979]
# runing time:34min10s