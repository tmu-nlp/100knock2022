#学習
GPU1=$1 \
GPU2=$2 \
CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 fairseq-train data/data-bin/kftt.ja-en \
    --save-dir checkpoints96/kftt.ja-en/ \
    --tensorboard-logdir log96 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 10