GPU=$1
BIN=/home/kyotaro/100knock/preprocess-kyoto

CUDA_VISIBLE_DEVICES=$GPU fairseq-train $BIN \
    --seed 1 \
    --keep-last-epochs 1 \
    --arch transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 1000 \
    --lr 0.0005 \
    --update-freq 8 \
    --max-epoch 5 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 2048 \
    --source-lang ja \
    --target-lang en \
    --save-dir checkpoints-kyoto > train.log
# CUDA_VISIBLE_DEVICES=$GPU fairseq-train $BIN \
#     --fp16 \
#     --save-dir save91 \
#     --max-epoch 10 \
#     --arch transformer --share-decoder-input-output-embed \
#     --optimizer adam --clip-norm 1.0 \
#     --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
#     --update-freq 1 \
#     --dropout 0.2 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 8000 > 91.log