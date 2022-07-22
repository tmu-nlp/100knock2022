echo 'binarize data'
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref train.sub \
    --validpref dev.sub \
    --srcdict preprocess-kyoto-98-jp/dict.ja.txt \
    --tgtdict preprocess-kyoto-98-jp/dict.en.txt \
    --destdir preprocess-kyoto-98-kftt

GPU=$1
BIN=preprocess-kyoto-98-kftt

CUDA_VISIBLE_DEVICES=$GPU fairseq-train $BIN \
    --seed 1 \
    --keep-last-epochs 1 \
    --arch transformer \
    --optimizer adam \
    --restore-file checkpoints-kyoto-98-jp/checkpoint5.pt
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
    --save-dir checkpoints-kyoto-98-jp > train98.log