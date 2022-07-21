#train
GPU=$1
CUDA_VISIBLE_DEVICES=$GPU fairseq-train data/data-bin/kftt.ja-en \
    --save-dir checkpoints/kftt.ja-en/ \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 10 \
| tee checkpoints/train.login